"""Deterministic, side-effect-free failure simulator for retry lessons."""


_attempts: dict[str, int] = {}
_always_timeout = False


def run_unstable_operation(task: str) -> str:
    """Fail twice with TimeoutError, then return a deterministic result."""
    attempt = _attempts.get(task, 0) + 1
    _attempts[task] = attempt

    if _always_timeout or attempt <= 2:
        raise TimeoutError(f"模拟临时超时：task={task}, attempt={attempt}")

    return f"unstable_tool 执行成功：{task}（第 {attempt} 次尝试）"


def reset_unstable_tool() -> None:
    """Reset deterministic counters between teaching scenarios and tests."""
    global _always_timeout
    _attempts.clear()
    _always_timeout = False


def set_unstable_always_timeout(enabled: bool) -> None:
    """Force every attempt to time out for the human-review scenario."""
    global _always_timeout
    _always_timeout = enabled


def get_unstable_attempts(task: str) -> int:
    """Return the execution count without changing tool behavior."""
    return _attempts.get(task, 0)
