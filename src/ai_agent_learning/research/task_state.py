from dataclasses import replace
from datetime import datetime, timezone

from ai_agent_learning.research.models import (
    ResearchTask,
    ResearchTaskStatus,
)


MAX_TASK_TRANSITION_REASON_LENGTH = 2_000
MAX_TASK_RESULT_SUMMARY_LENGTH = 10_000
LEGAL_TASK_TRANSITIONS: dict[ResearchTaskStatus, frozenset[ResearchTaskStatus]] = {
    "pending": frozenset({"running", "cancelled"}),
    "running": frozenset({"blocked", "completed", "failed", "cancelled"}),
    "blocked": frozenset({"running", "cancelled"}),
    "failed": frozenset({"pending"}),
    "completed": frozenset(),
    "cancelled": frozenset(),
}


class TaskTransitionError(ValueError):
    """Base error raised by deterministic task transition policy."""


class InvalidTaskTransitionError(TaskTransitionError):
    pass


class InvalidTaskTransitionInputError(TaskTransitionError):
    pass


def transition_task_state(
    task: ResearchTask,
    target_status: ResearchTaskStatus,
    *,
    reason: str | None = None,
    result_summary: str | None = None,
    timestamp: str | None = None,
) -> ResearchTask:
    """Apply exactly one legal transition without I/O or model calls."""
    if target_status not in LEGAL_TASK_TRANSITIONS[task.status]:
        raise InvalidTaskTransitionError(
            f"不允许从 {task.status} 转换到 {target_status}"
        )

    normalized_reason = _optional_limited_text(
        reason,
        "reason",
        MAX_TASK_TRANSITION_REASON_LENGTH,
    )
    normalized_summary = _optional_limited_text(
        result_summary,
        "result_summary",
        MAX_TASK_RESULT_SUMMARY_LENGTH,
    )
    if target_status in {"blocked", "failed"} and not normalized_reason:
        raise InvalidTaskTransitionInputError(
            f"转换到 {target_status} 时必须提供 reason"
        )
    if target_status != "completed" and normalized_summary is not None:
        raise InvalidTaskTransitionInputError(
            "result_summary 只允许在转换到 completed 时提供"
        )
    if target_status not in {"blocked", "failed"} and normalized_reason:
        raise InvalidTaskTransitionInputError(
            "reason 只允许在转换到 blocked 或 failed 时提供"
        )

    now = timestamp or datetime.now(timezone.utc).isoformat()
    changes: dict[str, object] = {
        "status": target_status,
        "updated_at": now,
    }

    if task.status == "pending" and target_status == "running":
        changes.update(
            started_at=task.started_at or now,
            completed_at=None,
            error_message=None,
        )
    elif task.status == "blocked" and target_status == "running":
        changes.update(error_message=None)
    elif target_status == "blocked":
        changes.update(error_message=normalized_reason)
    elif target_status == "completed":
        changes.update(
            completed_at=now,
            error_message=None,
            result_summary=normalized_summary,
        )
    elif target_status == "failed":
        changes.update(
            completed_at=now,
            error_message=normalized_reason,
            result_summary=None,
        )
    elif target_status == "cancelled":
        changes.update(completed_at=now, error_message=None)
    elif task.status == "failed" and target_status == "pending":
        # A retry is a new execution attempt, so its next running transition
        # receives a new started_at rather than reusing the failed attempt.
        changes.update(
            started_at=None,
            completed_at=None,
            error_message=None,
            result_summary=None,
        )

    return replace(task, **changes)


def _optional_limited_text(
    value: str | None,
    field_name: str,
    max_length: int,
) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        return None
    if len(normalized) > max_length:
        raise InvalidTaskTransitionInputError(
            f"{field_name} 不能超过 {max_length} 个字符"
        )
    return normalized
