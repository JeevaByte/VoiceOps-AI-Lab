import asyncio

from services.gateway.app.telemetry import TelemetryPublisher


class _FakeRedisClient:
    def __init__(self):
        self.calls = []

    async def publish(self, channel, payload):
        self.calls.append((channel, payload))


def test_publish_no_redis_url_is_noop():
    pub = TelemetryPublisher(redis_url=None, channel="events")
    asyncio.run(pub.publish({"a": 1}))
    assert pub._client is None


def test_publish_initializes_client_and_publishes(monkeypatch):
    import redis.asyncio as redis

    fake_client = _FakeRedisClient()
    monkeypatch.setattr(redis, "from_url", lambda *args, **kwargs: fake_client)

    pub = TelemetryPublisher(redis_url="redis://localhost:6379/0", channel="events")
    asyncio.run(pub.publish({"hello": "world"}))
    assert pub._client is fake_client
    assert len(fake_client.calls) == 1
    assert fake_client.calls[0][0] == "events"
    assert '"hello": "world"' in fake_client.calls[0][1]


def test_publish_handles_client_init_failure(monkeypatch):
    import redis.asyncio as redis

    def boom(*args, **kwargs):
        raise RuntimeError("cannot connect")

    monkeypatch.setattr(redis, "from_url", boom)
    pub = TelemetryPublisher(redis_url="redis://localhost:6379/0", channel="events")
    asyncio.run(pub.publish({"x": 1}))
    assert pub._client is None


def test_publish_handles_publish_failure(monkeypatch):
    import redis.asyncio as redis

    class _FailingClient:
        async def publish(self, channel, payload):
            raise RuntimeError("publish failed")

    monkeypatch.setattr(redis, "from_url", lambda *args, **kwargs: _FailingClient())
    pub = TelemetryPublisher(redis_url="redis://localhost:6379/0", channel="events")
    asyncio.run(pub.publish({"x": 1}))
