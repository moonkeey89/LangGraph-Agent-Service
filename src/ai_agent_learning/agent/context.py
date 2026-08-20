from dataclasses import dataclass


@dataclass(frozen=True)
class AgentContext:
    """Trusted, run-scoped values supplied by the application."""

    user_id: str

