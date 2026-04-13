from __future__ import annotations

import json
from typing import AsyncIterator

import httpx


class STTClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    async def transcribe(self, pcm16_b64: str) -> dict:
        async with httpx.AsyncClient(timeout=60) as client:
            res = await client.post(
                f"{self.base_url}/transcribe",
                json={"pcm16_b64": pcm16_b64, "sample_rate": 16000, "chunk_ms": 1200},
            )
            res.raise_for_status()
            return res.json()


class LLMClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST", f"{self.base_url}/generate_stream", json={"prompt": prompt}
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    delta = data.get("delta", "")
                    if delta:
                        yield delta
                    if data.get("done"):
                        break


class TTSClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    async def synthesize(self, text: str) -> dict:
        async with httpx.AsyncClient(timeout=120) as client:
            res = await client.post(f"{self.base_url}/synthesize", json={"text": text})
            res.raise_for_status()
            return res.json()
