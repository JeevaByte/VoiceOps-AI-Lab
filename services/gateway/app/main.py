from __future__ import annotations

import base64
import json
import uuid
from collections import deque
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from .clients import LLMClient, STTClient, TTSClient
from .codec import AudioCodec, CodecError
from .config import settings
from .state import SessionState, StateMachine
from .telemetry import TelemetryPublisher

app = FastAPI(title="voiceops-gateway", version="0.1.0")


class SessionContext:
    def __init__(self) -> None:
        self.id = str(uuid.uuid4())
        self.state = StateMachine()
        self.audio_buffer = deque()
        self.audio_size = 0
        self.dropped_frames = 0

    def push_audio(self, frame: bytes, max_buffer_bytes: int, max_frame_bytes: int) -> bool:
        if len(frame) > max_frame_bytes:
            self.dropped_frames += 1
            return False
        if self.audio_size + len(frame) > max_buffer_bytes:
            self.dropped_frames += 1
            return False
        self.audio_buffer.append(frame)
        self.audio_size += len(frame)
        return True

    def consume_audio(self) -> bytes:
        out = b"".join(self.audio_buffer)
        self.audio_buffer.clear()
        self.audio_size = 0
        return out


def chunk_bytes(data: bytes, size: int = 4096) -> list[bytes]:
    return [data[i : i + size] for i in range(0, len(data), size)]


codec = AudioCodec(
    backend=settings.codec_backend,
    input_format=settings.codec_input_format,
    allow_passthrough=settings.codec_allow_passthrough,
)
stt_client = STTClient(settings.stt_url)
llm_client = LLMClient(settings.llm_url)
tts_client = TTSClient(settings.tts_url)
telemetry = TelemetryPublisher(settings.redis_url, settings.telemetry_channel)


async def send_event(ws: WebSocket, event_type: str, payload: dict[str, Any]) -> None:
    await ws.send_text(json.dumps({"type": event_type, **payload}))


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.websocket("/ws/session")
async def ws_session(ws: WebSocket) -> None:
    await ws.accept()
    session = SessionContext()
    await send_event(ws, "session.started", {"session_id": session.id, "state": session.state.current.value})
    await telemetry.publish({"session_id": session.id, "event": "session.started"})
    try:
        while True:
            msg = await ws.receive()
            if "bytes" in msg and msg["bytes"] is not None:
                data = msg["bytes"]
                if not session.state.can_accept_audio():
                    await send_event(ws, "audio.ignored", {"reason": "half_duplex", "state": session.state.current.value})
                    continue
                accepted = session.push_audio(data, settings.max_buffer_bytes, settings.max_frame_bytes)
                if not accepted:
                    await send_event(
                        ws,
                        "backpressure.drop",
                        {"dropped_frames": session.dropped_frames, "buffer_bytes": session.audio_size},
                    )
                continue

            if "text" in msg and msg["text"] is not None:
                payload = json.loads(msg["text"])
                evt = payload.get("type")
                if evt == "input_audio_end":
                    await process_turn(ws, session)
                elif evt == "ping":
                    await send_event(ws, "pong", {})
                else:
                    await send_event(ws, "error", {"message": f"unknown event: {evt}"})
    except WebSocketDisconnect:
        await telemetry.publish({"session_id": session.id, "event": "session.disconnected"})


async def process_turn(ws: WebSocket, session: SessionContext) -> None:
    if session.state.current != SessionState.LISTENING:
        return
    encoded_audio = session.consume_audio()
    if not encoded_audio:
        await send_event(ws, "warning", {"message": "no_audio_buffered"})
        return

    session.state.transition(SessionState.THINKING)
    await send_event(ws, "state.changed", {"state": session.state.current.value})
    await telemetry.publish({"session_id": session.id, "event": "state.changed", "state": session.state.current.value})

    try:
        pcm16 = codec.decode_opus_to_pcm16_16khz(encoded_audio)
    except CodecError as exc:
        session.state.transition(SessionState.LISTENING)
        await send_event(ws, "error", {"stage": "decode", "message": str(exc)})
        await send_event(ws, "state.changed", {"state": session.state.current.value})
        return

    stt_response = await stt_client.transcribe(base64.b64encode(pcm16).decode("ascii"))
    partials = stt_response.get("partials", [])
    for idx, part in enumerate(partials):
        await send_event(ws, "stt.partial", {"index": idx, "text": part})
    transcript = stt_response.get("final", "").strip()
    await send_event(ws, "stt.final", {"text": transcript})

    llm_text_parts = []
    async for delta in llm_client.stream(transcript):
        llm_text_parts.append(delta)
        await send_event(ws, "llm.delta", {"text": delta})
    llm_text = "".join(llm_text_parts).strip() or "I heard you."

    session.state.transition(SessionState.SPEAKING)
    await send_event(ws, "state.changed", {"state": session.state.current.value})
    await telemetry.publish({"session_id": session.id, "event": "state.changed", "state": session.state.current.value})

    tts_response = await tts_client.synthesize(llm_text)
    tts_pcm = base64.b64decode(tts_response["pcm16_b64"])
    tts_rate = int(tts_response.get("sample_rate", 16000))
    opus = codec.encode_pcm16_to_opus(tts_pcm, input_rate=tts_rate)

    for seq, frame in enumerate(chunk_bytes(opus, size=4096)):
        await send_event(
            ws,
            "tts.audio",
            {
                "seq": seq,
                "codec": "audio/ogg;codecs=opus",
                "sample_rate": 16000,
                "audio_b64": base64.b64encode(frame).decode("ascii"),
            },
        )

    await send_event(ws, "tts.end", {"chars": len(llm_text)})
    session.state.transition(SessionState.LISTENING)
    await send_event(ws, "state.changed", {"state": session.state.current.value})
    await telemetry.publish({"session_id": session.id, "event": "state.changed", "state": session.state.current.value})
