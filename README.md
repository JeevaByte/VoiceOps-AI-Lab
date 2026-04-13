# VoiceOps-AI-Lab

Initial end-to-end **CPU-first, half-duplex Voice AI realtime pipeline**:

Client (Browser) → Gateway (WebSocket) → STT (Whisper wrapper) → LLM (Ollama wrapper) → TTS (Coqui wrapper) → Gateway → Client

## Architecture highlights

- **Half-duplex stability mode** implemented in gateway state machine:
  - `LISTENING` → accepts user audio frames
  - `THINKING` → STT + LLM
  - `SPEAKING` → sends TTS response audio, ignores new user audio (`audio.ignored`)
- **Audio path**
  - Browser input: Opus (`audio/webm;codecs=opus`)
  - Gateway decodes to **PCM16 mono 16kHz** before STT
  - TTS returns PCM16, gateway encodes to Opus (`audio/ogg;codecs=opus`) for client
- **CPU-first defaults**
  - STT defaults: `cpu` + `int8`
  - TTS defaults: `cpu`
  - LLM via Ollama HTTP (local)
- **Redis optional**
  - Used only for async telemetry publishing (`voiceops.session.events`), not in hot path

## Repository layout

- `/services/gateway` - FastAPI WebSocket gateway + state machine + backpressure + codec bridge
- `/services/stt` - Whisper wrapper service (`/transcribe`)
- `/services/llm` - Ollama wrapper service (`/generate_stream`)
- `/services/tts` - Coqui wrapper service (`/synthesize`)
- `/client/index.html` - minimal browser demo client
- `/tests/gateway` - focused tests for state and framing
- `/k8s/*.yaml` - Kubernetes manifests per service
- `/docker-compose.yaml` - local multi-service stack

## WebSocket protocol (gateway)

Endpoint: `ws://<host>:8000/ws/session`

### Client → Gateway
- **binary frame**: Opus audio chunk (browser MediaRecorder chunk)
- JSON text events:
  - `{"type":"input_audio_end"}` - finalize current utterance and run STT→LLM→TTS turn
  - `{"type":"ping"}`

### Gateway → Client events
- `session.started`
- `state.changed` (`LISTENING|THINKING|SPEAKING`)
- `stt.partial`
- `stt.final`
- `llm.delta`
- `tts.audio` (base64 Opus chunk, `audio/ogg;codecs=opus`)
- `tts.end`
- `audio.ignored` (half-duplex enforcement)
- `backpressure.drop`
- `error`, `warning`, `pong`

## Backpressure behavior

Gateway uses configurable limits:
- `GATEWAY_MAX_FRAME_BYTES` (default `48000`)
- `GATEWAY_MAX_BUFFER_BYTES` (default `512000`)

When limits are exceeded, new frames are dropped and `backpressure.drop` is emitted.

## Local run (Docker Compose)

```bash
docker compose up --build
```

Services:
- gateway: `:8000`
- stt: `:8001`
- llm: `:8002`
- tts: `:8003`
- redis: `:6379`
- ollama: `:11434`

Then open `/client/index.html` (or host it from a static file server) and connect to gateway.

## Run services manually (dev)

```bash
# gateway
uvicorn services.gateway.app.main:app --host 0.0.0.0 --port 8000
# stt
uvicorn services.stt.app.main:app --host 0.0.0.0 --port 8001
# llm
uvicorn services.llm.app.main:app --host 0.0.0.0 --port 8002
# tts
uvicorn services.tts.app.main:app --host 0.0.0.0 --port 8003
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q tests/gateway
```

## Kubernetes deployment

Apply manifests:

```bash
kubectl apply -f k8s/redis.yaml
kubectl apply -f k8s/stt.yaml
kubectl apply -f k8s/llm.yaml
kubectl apply -f k8s/tts.yaml
kubectl apply -f k8s/gateway.yaml
```

Included resources:
- Deployments + Services for each component
- ConfigMap placeholders for service configuration
- Gateway HPA example (CPU)

### HPA notes
- Scale gateway primarily on CPU and connection metrics.
- Keep STT/LLM/TTS CPU-first initially; add GPU node selectors/tolerations later.
- Redis remains optional and out of hot path.

## Config knobs for future GPU upgrades

- STT: `STT_WHISPER_DEVICE` (`cpu` → `cuda`) and compute type
- TTS: `TTS_DEVICE` (`cpu` → `cuda`)
- LLM: switch Ollama model and host endpoint

## Codec note

Gateway uses ffmpeg subprocess for Opus decode/encode by default:
- `GATEWAY_CODEC_BACKEND=ffmpeg`
- `GATEWAY_CODEC_INPUT_FORMAT=webm`

If browser/container codec specifics differ, adjust input format accordingly.
