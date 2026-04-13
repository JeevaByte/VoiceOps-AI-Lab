from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    stt_url: str = "http://stt:8001"
    llm_url: str = "http://llm:8002"
    tts_url: str = "http://tts:8003"
    redis_url: str | None = None
    telemetry_channel: str = "voiceops.session.events"
    max_buffer_bytes: int = 512_000
    max_frame_bytes: int = 48_000
    codec_backend: str = "ffmpeg"
    codec_input_format: str = "webm"
    codec_allow_passthrough: bool = False

    model_config = SettingsConfigDict(env_prefix="GATEWAY_", env_file=".env", extra="ignore")


settings = Settings()
