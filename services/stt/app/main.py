from __future__ import annotations

import base64
import os
from dataclasses import dataclass

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="voiceops-stt", version="0.1.0")


class TranscribeRequest(BaseModel):
    pcm16_b64: str
    sample_rate: int = 16000
    chunk_ms: int = 1200


class TranscribeResponse(BaseModel):
    partials: list[str]
    final: str


@dataclass
class WhisperConfig:
    model_size: str = os.getenv("STT_WHISPER_MODEL", "small")
    device: str = os.getenv("STT_WHISPER_DEVICE", "cpu")
    compute_type: str = os.getenv("STT_WHISPER_COMPUTE_TYPE", "int8")


class WhisperEngine:
    def __init__(self, cfg: WhisperConfig):
        self.cfg = cfg
        self._model = None

    def _get_model(self):
        if self._model is not None:
            return self._model
        try:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self.cfg.model_size,
                device=self.cfg.device,
                compute_type=self.cfg.compute_type,
            )
        except Exception:
            self._model = False
        return self._model

    def transcribe(self, pcm16: bytes, sample_rate: int, chunk_ms: int) -> TranscribeResponse:
        if not pcm16:
            return TranscribeResponse(partials=[], final="")
        audio = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        samples_per_chunk = max(1, int(sample_rate * (chunk_ms / 1000.0)))
        chunks = [audio[i : i + samples_per_chunk] for i in range(0, len(audio), samples_per_chunk)]

        model = self._get_model()
        partials: list[str] = []
        for chunk in chunks:
            if len(chunk) == 0:
                continue
            if model is False:
                partials.append("[whisper-unavailable]")
                continue
            segments, _ = model.transcribe(chunk, beam_size=1, language="en")
            text = " ".join(seg.text.strip() for seg in segments).strip()
            if text:
                partials.append(text)

        final = " ".join(p for p in partials if p and p != "[whisper-unavailable]").strip()
        if not final and partials:
            final = "audio_received"
        return TranscribeResponse(partials=partials, final=final)


engine = WhisperEngine(WhisperConfig())


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/transcribe", response_model=TranscribeResponse)
def transcribe(req: TranscribeRequest) -> TranscribeResponse:
    pcm16 = base64.b64decode(req.pcm16_b64)
    return engine.transcribe(pcm16=pcm16, sample_rate=req.sample_rate, chunk_ms=req.chunk_ms)
