from __future__ import annotations

import json
import os
from typing import AsyncGenerator

import httpx
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI(title="voiceops-llm", version="0.1.0")

OLLAMA_URL = os.getenv("LLM_OLLAMA_URL", "http://ollama:11434")
OLLAMA_MODEL = os.getenv("LLM_MODEL", "llama3.2:1b")


class GenerateRequest(BaseModel):
    prompt: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/generate_stream")
async def generate_stream(req: GenerateRequest) -> StreamingResponse:
    async def stream() -> AsyncGenerator[str, None]:
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                async with client.stream(
                    "POST",
                    f"{OLLAMA_URL}/api/generate",
                    json={"model": OLLAMA_MODEL, "prompt": req.prompt, "stream": True},
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        data = json.loads(line)
                        yield json.dumps(
                            {
                                "delta": data.get("response", ""),
                                "done": bool(data.get("done", False)),
                            }
                        ) + "\n"
        except Exception:
            fallback = f"Echo: {req.prompt}".strip()
            for token in fallback.split():
                yield json.dumps({"delta": token + " ", "done": False}) + "\n"
            yield json.dumps({"delta": "", "done": True}) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")
