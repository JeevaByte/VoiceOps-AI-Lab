from services.gateway.app.main import SessionContext, chunk_bytes
from services.gateway.app.state import SessionState


def test_chunk_bytes():
    payload = b"a" * 9000
    chunks = chunk_bytes(payload, size=4096)
    assert [len(c) for c in chunks] == [4096, 4096, 808]


def test_backpressure_and_half_duplex_behavior():
    ctx = SessionContext()

    accepted = ctx.push_audio(b"x" * 400, max_buffer_bytes=500, max_frame_bytes=500)
    assert accepted is True

    dropped = ctx.push_audio(b"y" * 200, max_buffer_bytes=500, max_frame_bytes=500)
    assert dropped is False
    assert ctx.dropped_frames == 1

    ctx.state.transition(SessionState.THINKING)
    assert ctx.state.can_accept_audio() is False
