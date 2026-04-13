import asyncio

from services.gateway.app.clients import LLMClient, STTClient, TTSClient


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _StreamResponse:
    def __init__(self, lines):
        self._lines = lines

    def raise_for_status(self):
        return None

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class _StreamCM:
    def __init__(self, response):
        self.response = response

    async def __aenter__(self):
        return self.response

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _ClientCM:
    def __init__(self, response=None, stream_response=None):
        self.response = response
        self.stream_response = stream_response
        self.post_calls = []
        self.stream_calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, json):
        self.post_calls.append((url, json))
        return self.response

    def stream(self, method, url, json):
        self.stream_calls.append((method, url, json))
        return _StreamCM(self.stream_response)


def test_stt_client_transcribe_posts_expected_payload(monkeypatch):
    fake_client = _ClientCM(response=_Response({"final": "hi"}))

    from services.gateway.app import clients as clients_module

    monkeypatch.setattr(clients_module.httpx, "AsyncClient", lambda timeout: fake_client)
    client = STTClient("http://stt:8001/")
    out = asyncio.run(client.transcribe("abc"))
    assert out == {"final": "hi"}
    assert fake_client.post_calls[0][0] == "http://stt:8001/transcribe"


def test_tts_client_synthesize_posts_expected_payload(monkeypatch):
    fake_client = _ClientCM(response=_Response({"pcm16_b64": "QQ==", "sample_rate": 16000}))

    from services.gateway.app import clients as clients_module

    monkeypatch.setattr(clients_module.httpx, "AsyncClient", lambda timeout: fake_client)
    client = TTSClient("http://tts:8003/")
    out = asyncio.run(client.synthesize("hello"))
    assert out["sample_rate"] == 16000
    assert fake_client.post_calls[0][0] == "http://tts:8003/synthesize"
    assert fake_client.post_calls[0][1] == {"text": "hello"}


def test_llm_client_stream_yields_only_non_empty_deltas(monkeypatch):
    fake_stream = _StreamResponse(
        [
            "",
            '{"delta":"hel","done":false}',
            '{"delta":"","done":false}',
            '{"delta":"lo","done":false}',
            '{"delta":"","done":true}',
            '{"delta":"ignored","done":false}',
        ]
    )
    fake_client = _ClientCM(stream_response=fake_stream)

    from services.gateway.app import clients as clients_module

    monkeypatch.setattr(clients_module.httpx, "AsyncClient", lambda timeout: fake_client)
    client = LLMClient("http://llm:8002/")

    async def _collect():
        parts = []
        async for delta in client.stream("prompt"):
            parts.append(delta)
        return parts

    out = asyncio.run(_collect())
    assert out == ["hel", "lo"]
    assert fake_client.stream_calls[0][1] == "http://llm:8002/generate_stream"
