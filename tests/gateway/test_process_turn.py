import asyncio
import base64
import json

from services.gateway.app.codec import CodecError
from services.gateway.app.main import SessionContext, process_turn
from services.gateway.app.state import SessionState


class _FakeWS:
    pass


def test_process_turn_noop_when_not_listening(monkeypatch):
    from services.gateway.app import main as gateway_main

    events = []

    async def fake_send_event(ws, event_type, payload):
        events.append((event_type, payload))

    monkeypatch.setattr(gateway_main, "send_event", fake_send_event)

    session = SessionContext()
    session.state.transition(SessionState.THINKING)
    asyncio.run(process_turn(_FakeWS(), session))
    assert events == []


def test_process_turn_warns_when_no_audio(monkeypatch):
    from services.gateway.app import main as gateway_main

    events = []

    async def fake_send_event(ws, event_type, payload):
        events.append((event_type, payload))

    monkeypatch.setattr(gateway_main, "send_event", fake_send_event)

    session = SessionContext()
    asyncio.run(process_turn(_FakeWS(), session))
    assert events == [("warning", {"message": "no_audio_buffered"})]


def test_process_turn_decode_error_resets_state(monkeypatch):
    from services.gateway.app import main as gateway_main

    events = []

    async def fake_send_event(ws, event_type, payload):
        events.append((event_type, payload))

    class _Codec:
        def decode_opus_to_pcm16_16khz(self, encoded):
            raise CodecError("bad decode")

    monkeypatch.setattr(gateway_main, "send_event", fake_send_event)
    monkeypatch.setattr(gateway_main, "codec", _Codec())

    session = SessionContext()
    assert session.push_audio(b"abc", max_buffer_bytes=100, max_frame_bytes=100)
    asyncio.run(process_turn(_FakeWS(), session))

    assert session.state.current == SessionState.LISTENING
    assert events[0][0] == "state.changed"
    assert events[0][1]["state"] == "THINKING"
    assert events[1] == ("error", {"stage": "decode", "message": "bad decode"})
    assert events[2] == ("state.changed", {"state": "LISTENING"})


def test_process_turn_happy_path_emits_expected_events(monkeypatch):
    from services.gateway.app import main as gateway_main

    events = []

    async def fake_send_event(ws, event_type, payload):
        events.append((event_type, payload))

    class _Codec:
        def decode_opus_to_pcm16_16khz(self, encoded):
            return b"\x00\x01"

        def encode_pcm16_to_opus(self, pcm16, input_rate=16000):
            return b"x" * 5000

    class _STT:
        async def transcribe(self, pcm16_b64):
            assert base64.b64decode(pcm16_b64) == b"\x00\x01"
            return {"partials": ["hi"], "final": "hello"}

    class _LLM:
        async def stream(self, prompt):
            for part in ["a", "b"]:
                yield part

    class _TTS:
        async def synthesize(self, text):
            assert text == "ab"
            return {"pcm16_b64": base64.b64encode(b"\x00\x00").decode("ascii"), "sample_rate": 16000}

    class _Telemetry:
        async def publish(self, event):
            return None

    monkeypatch.setattr(gateway_main, "send_event", fake_send_event)
    monkeypatch.setattr(gateway_main, "codec", _Codec())
    monkeypatch.setattr(gateway_main, "stt_client", _STT())
    monkeypatch.setattr(gateway_main, "llm_client", _LLM())
    monkeypatch.setattr(gateway_main, "tts_client", _TTS())
    monkeypatch.setattr(gateway_main, "telemetry", _Telemetry())

    session = SessionContext()
    assert session.push_audio(b"abc", max_buffer_bytes=100, max_frame_bytes=100)
    asyncio.run(process_turn(_FakeWS(), session))

    event_types = [etype for etype, _ in events]
    assert event_types.count("state.changed") == 3
    assert "stt.partial" in event_types
    assert "stt.final" in event_types
    assert event_types.count("llm.delta") == 2
    assert event_types.count("tts.audio") == 2
    assert "tts.end" in event_types
    assert session.state.current == SessionState.LISTENING

    final_state_events = [payload for etype, payload in events if etype == "state.changed"]
    assert final_state_events[-1]["state"] == "LISTENING"

    tts_audio_payloads = [payload for etype, payload in events if etype == "tts.audio"]
    assert tts_audio_payloads[0]["seq"] == 0
    assert tts_audio_payloads[1]["seq"] == 1

    # Confirm serialized content is JSON-compatible for event payloads
    json.dumps(events)
