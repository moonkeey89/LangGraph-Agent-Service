import re
from dataclasses import replace
from datetime import datetime, timezone

from ai_agent_learning.research.models import (
    AgentRun,
    AgentRunOutcome,
    AgentRunStatus,
)


MAX_RUN_ERROR_LENGTH = 2_000
LEGAL_RUN_TRANSITIONS: dict[AgentRunStatus, frozenset[AgentRunStatus]] = {
    "pending": frozenset({"running", "failed", "cancelled"}),
    "running": frozenset({"interrupted", "completed", "failed", "cancelled"}),
    "interrupted": frozenset({"running", "failed", "cancelled"}),
    "completed": frozenset(),
    "failed": frozenset(),
    "cancelled": frozenset(),
}
_CREDENTIAL_PATTERN = re.compile(
    r"(?i)\b(api[_-]?key|password|passwd|token|secret)\b\s*[:=]\s*\S+"
)
_OPENAI_STYLE_KEY_PATTERN = re.compile(r"\bsk-[A-Za-z0-9_-]{6,}\b")


class AgentRunTransitionError(ValueError):
    """Base error raised by deterministic AgentRun lifecycle policy."""


class InvalidAgentRunTransitionError(AgentRunTransitionError):
    pass


class InvalidAgentRunTransitionInputError(AgentRunTransitionError):
    pass


def transition_agent_run(
    run: AgentRun,
    target_status: AgentRunStatus,
    *,
    error_message: str | None = None,
    timestamp: str | None = None,
) -> AgentRun:
    """Apply one legal AgentRun transition without performing I/O."""
    if target_status not in LEGAL_RUN_TRANSITIONS[run.status]:
        raise InvalidAgentRunTransitionError(
            f"不允许从 {run.status} 转换到 {target_status}"
        )

    cleaned_error = clean_run_error(error_message)
    if target_status == "failed" and cleaned_error is None:
        raise InvalidAgentRunTransitionInputError(
            "转换到 failed 时必须提供 error_message"
        )
    if target_status != "failed" and cleaned_error is not None:
        raise InvalidAgentRunTransitionInputError(
            "error_message 只允许在转换到 failed 时提供"
        )
    if target_status == "completed" and run.output_artifact_id is None:
        raise InvalidAgentRunTransitionInputError(
            "完成运行前必须绑定 output_artifact_id"
        )

    now = timestamp or datetime.now(timezone.utc).isoformat()
    changes: dict[str, object] = {
        "status": target_status,
        "updated_at": now,
        "error_message": cleaned_error if target_status == "failed" else None,
    }
    if run.status == "pending" and target_status == "running":
        changes["started_at"] = run.started_at or now
    if target_status == "interrupted":
        changes["finished_at"] = None
    elif target_status in {"completed", "failed", "cancelled"}:
        changes["finished_at"] = now
    elif run.status == "interrupted" and target_status == "running":
        changes["started_at"] = run.started_at
        changes["finished_at"] = None

    return replace(run, **changes)


def finalize_agent_run(
    run: AgentRun,
    *,
    outcome: AgentRunOutcome,
    output_artifact_id: str | None,
    message: str | None = None,
    timestamp: str | None = None,
) -> AgentRun:
    """Map one terminal Research Graph outcome to durable AgentRun metadata."""
    if run.status != "running":
        raise InvalidAgentRunTransitionError(
            f"只有 running Run 可以收尾，当前状态为 {run.status}"
        )
    if outcome in {"completed", "needs_review"} and not output_artifact_id:
        raise InvalidAgentRunTransitionInputError(
            f"{outcome} 结果必须关联 output_artifact_id"
        )
    if outcome in {"blocked", "failed"} and output_artifact_id is not None:
        raise InvalidAgentRunTransitionInputError(
            f"{outcome} 结果不能关联 output_artifact_id"
        )
    cleaned_message = clean_run_error(message)
    if outcome in {"blocked", "failed", "needs_review"} and not cleaned_message:
        raise InvalidAgentRunTransitionInputError(
            f"{outcome} 结果必须提供安全说明"
        )

    now = timestamp or datetime.now(timezone.utc).isoformat()
    return replace(
        run,
        status="failed" if outcome == "failed" else "completed",
        outcome=outcome,
        output_artifact_id=output_artifact_id,
        error_message=cleaned_message,
        finished_at=now,
        updated_at=now,
    )


def clean_run_error(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = " ".join(value.strip().splitlines())
    if not normalized:
        return None
    normalized = _CREDENTIAL_PATTERN.sub(r"\1=[REDACTED]", normalized)
    normalized = _OPENAI_STYLE_KEY_PATTERN.sub("[REDACTED]", normalized)
    if normalized.lower().startswith("traceback"):
        normalized = "内部执行失败，详细堆栈已隐藏"
    return normalized[:MAX_RUN_ERROR_LENGTH]
