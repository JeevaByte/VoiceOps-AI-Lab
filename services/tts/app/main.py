from __future__ import annotations

import base64
import math
import os
import tempfile
import wave

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="voiceops-tts", version="0.1.0")


class SynthesizeRequest(BaseModel):
    text: str


class TTSResponse(BaseModel):
    pcm16_b64: str
    sample_rate: int


class CoquiEngine:
    def __init__(self):
        self.model_name = os.getenv("TTS_MODEL_NAME", "tts_models/en/ljspeech/tacotron2-DDC")
        self.device = os.getenv("TTS_DEVICE", "cpu")
        self._tts = None

    def _get_tts(self):
        if self._tts is not None:
            return self._tts
        try:
            from TTS.api import TTS

            self._tts = TTS(self.model_name).to(self.device)
        except Exception:
            self._tts = False
        return self._tts

    def synthesize(self, text: str) -> tuple[bytes, int]:
        tts = self._get_tts()
        if tts is False:
            return self._fallback_tone(max(0.4, min(2.5, len(text) * 0.02))), 16000

        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            tts.tts_to_file(text=text, file_path=tmp.name)
            with wave.open(tmp.name, "rb") as wav:
                sr = wav.getframerate()
                frames = wav.readframes(wav.getnframes())
                return frames, sr

    @staticmethod
    def _fallback_tone(duration_sec: float) -> bytes:
        sample_rate = 16000
        total = int(sample_rate * duration_sec)
        pcm = bytearray()
        for i in range(total):
            sample = int(6000 * math.sin(2 * math.pi * 220 * (i / sample_rate)))
            pcm.extend(int(sample).to_bytes(2, "little", signed=True))
        return bytes(pcm)


engine = CoquiEngine()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/synthesize", response_model=TTSResponse)
def synthesize(req: SynthesizeRequest) -> TTSResponse:
    pcm16, sample_rate = engine.synthesize(req.text)
    if sample_rate != 16000:
        audio = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32)
        x_old = np.linspace(0, 1, num=len(audio), endpoint=False)
        x_new = np.linspace(0, 1, num=max(1, int(len(audio) * 16000 / sample_rate)), endpoint=False)
        resampled = np.interp(x_new, x_old, audio).astype(np.int16)
        pcm16 = resampled.tobytes()
        sample_rate = 16000
    return TTSResponse(pcm16_b64=base64.b64encode(pcm16).decode("ascii"), sample_rate=sample_rate)
