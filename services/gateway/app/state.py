from enum import Enum


class SessionState(str, Enum):
    LISTENING = "LISTENING"
    THINKING = "THINKING"
    SPEAKING = "SPEAKING"


ALLOWED_TRANSITIONS = {
    SessionState.LISTENING: {SessionState.THINKING},
    SessionState.THINKING: {SessionState.SPEAKING, SessionState.LISTENING},
    SessionState.SPEAKING: {SessionState.LISTENING},
}


class StateMachine:
    def __init__(self) -> None:
        self.current = SessionState.LISTENING

    def transition(self, nxt: SessionState) -> SessionState:
        if nxt == self.current:
            return self.current
        if nxt not in ALLOWED_TRANSITIONS[self.current]:
            raise ValueError(f"Invalid transition: {self.current} -> {nxt}")
        self.current = nxt
        return self.current

    def can_accept_audio(self) -> bool:
        return self.current == SessionState.LISTENING
