import base64

import numpy as np

from services.stt.app import main as stt_main


def _pcm_from_int16(values):
    return np.array(values, dtype=np.int16).tobytes()


def test_health():
    assert stt_main.health() == {"status": "ok"}


def test_whisper_engine_empty_audio_returns_empty_result():
    engine = stt_main.WhisperEngine(stt_main.WhisperConfig())
    out = engine.transcribe(b"", sample_rate=16000, chunk_ms=1200)
    assert out.partials == []
    assert out.final == ""


def test_whisper_engine_unavailable_model_returns_fallback_marker(monkeypatch):
    engine = stt_main.WhisperEngine(stt_main.WhisperConfig())
    monkeypatch.setattr(engine, "_get_model", lambda: False)

    out = engine.transcribe(_pcm_from_int16([1, 2, 3, 4]), sample_rate=2, chunk_ms=500)
    assert out.partials
    assert all(part == "[whisper-unavailable]" for part in out.partials)
    assert out.final == "audio_received"


def test_whisper_engine_collects_transcript_from_segments(monkeypatch):
    class _Segment:
        def __init__(self, text):
            self.text = text

    class _Model:
        def transcribe(self, chunk, beam_size, language):
            return ([_Segment(" hello ")], None)

    engine = stt_main.WhisperEngine(stt_main.WhisperConfig())
    monkeypatch.setattr(engine, "_get_model", lambda: _Model())

    out = engine.transcribe(_pcm_from_int16([1, 2, 3, 4]), sample_rate=2, chunk_ms=500)
    assert out.partials == ["hello", "hello", "hello", "hello"]
    assert out.final == "hello hello hello hello"


def test_transcribe_endpoint_decodes_base64_and_calls_engine(monkeypatch):
    calls = {}

    class _Engine:
        def transcribe(self, pcm16, sample_rate, chunk_ms):
            calls["pcm16"] = pcm16
            calls["sample_rate"] = sample_rate
            calls["chunk_ms"] = chunk_ms
            return stt_main.TranscribeResponse(partials=["p"], final="f")

    monkeypatch.setattr(stt_main, "engine", _Engine())
    req = stt_main.TranscribeRequest(pcm16_b64=base64.b64encode(b"\x01\x02").decode("ascii"), sample_rate=8000, chunk_ms=500)
    out = stt_main.transcribe(req)

    assert out.partials == ["p"]
    assert out.final == "f"
    assert calls["pcm16"] == b"\x01\x02"
    assert calls["sample_rate"] == 8000
    assert calls["chunk_ms"] == 500
