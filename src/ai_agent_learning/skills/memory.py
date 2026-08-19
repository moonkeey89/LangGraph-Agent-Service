"""Teaching-only simulated memory write capability."""


_SAVED_MEMORIES: list[str] = []


def save_memory(content: str) -> str:
    """Save content in a process-local list to simulate a sensitive write."""
    _SAVED_MEMORIES.append(content)
    return f"已保存模拟记忆：{content}"


def get_saved_memories() -> tuple[str, ...]:
    """Return simulated memories for verification and teaching."""
    return tuple(_SAVED_MEMORIES)


def clear_saved_memories() -> None:
    """Reset simulated memories between tests."""
    _SAVED_MEMORIES.clear()
