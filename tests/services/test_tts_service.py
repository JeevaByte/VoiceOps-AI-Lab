import base64

import numpy as np

from services.tts.app import main as tts_main


def _pcm_from_int16(values):
    return np.array(values, dtype=np.int16).tobytes()


def test_health():
    assert tts_main.health() == {"status": "ok"}


def test_fallback_tone_generates_expected_pcm_size():
    pcm = tts_main.CoquiEngine._fallback_tone(0.5)
    # 16000 samples/sec * 0.5 sec * 2 bytes/sample
    assert len(pcm) == 16000


def test_engine_synthesize_uses_fallback_when_tts_unavailable(monkeypatch):
    engine = tts_main.CoquiEngine()
    monkeypatch.setattr(engine, "_get_tts", lambda: False)
    pcm, sr = engine.synthesize("hello")
    assert sr == 16000
    assert len(pcm) > 0


def test_synthesize_endpoint_resamples_to_16khz(monkeypatch):
    class _Engine:
        def synthesize(self, text):
            return _pcm_from_int16([0, 1000, -1000, 2000]), 8000

    monkeypatch.setattr(tts_main, "engine", _Engine())
    out = tts_main.synthesize(tts_main.SynthesizeRequest(text="test"))
    decoded = base64.b64decode(out.pcm16_b64)
    assert out.sample_rate == 16000
    assert len(decoded) > len(_pcm_from_int16([0, 1000, -1000, 2000]))


def test_synthesize_endpoint_keeps_16khz_unchanged(monkeypatch):
    raw = _pcm_from_int16([0, 1, 2, 3])

    class _Engine:
        def synthesize(self, text):
            return raw, 16000

    monkeypatch.setattr(tts_main, "engine", _Engine())
    out = tts_main.synthesize(tts_main.SynthesizeRequest(text="test"))
    assert out.sample_rate == 16000
    assert base64.b64decode(out.pcm16_b64) == raw
