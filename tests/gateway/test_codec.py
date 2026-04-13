import subprocess

from services.gateway.app.codec import AudioCodec, CodecError


def test_decode_empty_returns_empty_bytes():
    codec = AudioCodec()
    assert codec.decode_opus_to_pcm16_16khz(b"") == b""


def test_encode_empty_returns_empty_bytes():
    codec = AudioCodec()
    assert codec.encode_pcm16_to_opus(b"") == b""


def test_unsupported_backend_raises_when_passthrough_disabled():
    codec = AudioCodec(backend="unknown", allow_passthrough=False)
    try:
        codec.decode_opus_to_pcm16_16khz(b"abc")
        assert False, "Expected CodecError"
    except CodecError:
        assert True


def test_unsupported_backend_passthrough_returns_original():
    codec = AudioCodec(backend="unknown", allow_passthrough=True)
    payload = b"abc"
    assert codec.decode_opus_to_pcm16_16khz(payload) == payload
    assert codec.encode_pcm16_to_opus(payload) == payload


def test_run_success_returns_stdout(monkeypatch):
    codec = AudioCodec()

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=[], returncode=0, stdout=b"ok", stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    out = codec._run(["ffmpeg"], b"input", "decode")
    assert out == b"ok"


def test_run_failure_raises_codec_error(monkeypatch):
    codec = AudioCodec(allow_passthrough=False)

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=[], returncode=1, stdout=b"", stderr=b"boom")

    monkeypatch.setattr(subprocess, "run", fake_run)
    try:
        codec._run(["ffmpeg"], b"input", "decode")
        assert False, "Expected CodecError"
    except CodecError as exc:
        assert "boom" in str(exc)


def test_run_failure_returns_input_when_passthrough_enabled(monkeypatch):
    codec = AudioCodec(allow_passthrough=True)

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=[], returncode=1, stdout=b"", stderr=b"boom")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert codec._run(["ffmpeg"], b"input", "decode") == b"input"
