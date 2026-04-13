from __future__ import annotations

import json
from typing import Any


class TelemetryPublisher:
    def __init__(self, redis_url: str | None, channel: str):
        self._redis_url = redis_url
        self._channel = channel
        self._client = None

    async def publish(self, event: dict[str, Any]) -> None:
        if not self._redis_url:
            return
        if self._client is None:
            try:
                import redis.asyncio as redis  # type: ignore

                self._client = redis.from_url(self._redis_url, decode_responses=True)
            except Exception:
                return
        try:
            await self._client.publish(self._channel, json.dumps(event))
        except Exception:
            return
