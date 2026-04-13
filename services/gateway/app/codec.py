import subprocess


class CodecError(RuntimeError):
    pass


class AudioCodec:
    def __init__(self, backend: str = "ffmpeg", input_format: str = "webm", allow_passthrough: bool = False):
        self.backend = backend
        self.input_format = input_format
        self.allow_passthrough = allow_passthrough

    def decode_opus_to_pcm16_16khz(self, encoded: bytes) -> bytes:
        if not encoded:
            return b""
        if self.backend != "ffmpeg":
            if self.allow_passthrough:
                return encoded
            raise CodecError(f"Unsupported codec backend: {self.backend}")
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            self.input_format,
            "-i",
            "pipe:0",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "s16le",
            "pipe:1",
        ]
        return self._run(cmd, encoded, "decode")

    def encode_pcm16_to_opus(self, pcm16: bytes, input_rate: int = 16000) -> bytes:
        if not pcm16:
            return b""
        if self.backend != "ffmpeg":
            if self.allow_passthrough:
                return pcm16
            raise CodecError(f"Unsupported codec backend: {self.backend}")
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "s16le",
            "-ar",
            str(input_rate),
            "-ac",
            "1",
            "-i",
            "pipe:0",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "libopus",
            "-b:a",
            "24k",
            "-application",
            "voip",
            "-f",
            "ogg",
            "pipe:1",
        ]
        return self._run(cmd, pcm16, "encode")

    def _run(self, cmd: list[str], data: bytes, mode: str) -> bytes:
        proc = subprocess.run(cmd, input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if proc.returncode != 0:
            if self.allow_passthrough:
                return data
            raise CodecError(f"ffmpeg {mode} failed: {proc.stderr.decode('utf-8', errors='ignore')}")
        return proc.stdout
