from services.gateway.app.state import SessionState, StateMachine


def test_state_machine_happy_path():
    sm = StateMachine()
    assert sm.current == SessionState.LISTENING
    assert sm.can_accept_audio() is True

    sm.transition(SessionState.THINKING)
    assert sm.current == SessionState.THINKING
    assert sm.can_accept_audio() is False

    sm.transition(SessionState.SPEAKING)
    assert sm.current == SessionState.SPEAKING

    sm.transition(SessionState.LISTENING)
    assert sm.current == SessionState.LISTENING


def test_state_machine_rejects_invalid_transition():
    sm = StateMachine()
    try:
        sm.transition(SessionState.SPEAKING)
        assert False, "Expected ValueError"
    except ValueError:
        assert True
