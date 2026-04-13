import asyncio
import json

from services.llm.app import main as llm_main


def test_health():
    assert llm_main.health() == {"status": "ok"}


def _collect_streaming_body(response):
    async def _collect():
        chunks = []
        async for chunk in response.body_iterator:
            if isinstance(chunk, bytes):
                chunks.append(chunk.decode("utf-8"))
            else:
                chunks.append(chunk)
        return "".join(chunks)

    return asyncio.run(_collect())


def test_generate_stream_fallback_when_ollama_fails(monkeypatch):
    class _FailingClient:
        async def __aenter__(self):
            raise RuntimeError("network down")

        async def __aexit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(llm_main.httpx, "AsyncClient", lambda timeout: _FailingClient())
    response = asyncio.run(llm_main.generate_stream(llm_main.GenerateRequest(prompt="hello world")))
    body = _collect_streaming_body(response)
    lines = [json.loads(line) for line in body.strip().splitlines()]
    assert lines[-1] == {"delta": "", "done": True}
    assert any(item["delta"] for item in lines[:-1])


def test_generate_stream_uses_ollama_stream_when_available(monkeypatch):
    class _Resp:
        def raise_for_status(self):
            return None

        async def aiter_lines(self):
            for line in [
                "",
                '{"response":"hi ","done":false}',
                '{"response":"there","done":true}',
            ]:
                yield line

    class _RespCM:
        async def __aenter__(self):
            return _Resp()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def stream(self, method, url, json):
            return _RespCM()

    monkeypatch.setattr(llm_main.httpx, "AsyncClient", lambda timeout: _Client())
    response = asyncio.run(llm_main.generate_stream(llm_main.GenerateRequest(prompt="p")))
    body = _collect_streaming_body(response)
    lines = [json.loads(line) for line in body.strip().splitlines()]
    assert lines[0] == {"delta": "hi ", "done": False}
    assert lines[1] == {"delta": "there", "done": True}
