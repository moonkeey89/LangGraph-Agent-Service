import json
import logging
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Literal, Protocol

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command, Interrupt

from ai_agent_learning.agent.context import AgentContext


logger = logging.getLogger(__name__)


class CompiledGraph(Protocol):
    def invoke(self, input: Any, **kwargs: Any) -> Any: ...

    def get_state(self, config: dict[str, Any]) -> Any: ...


class AgentServiceError(RuntimeError):
    status_code = 400
    public_message = "Agent request could not be completed"


class PendingInterruptError(AgentServiceError):
    status_code = 409
    public_message = "This thread is interrupted; call the resume endpoint"


class NoPendingInterruptError(AgentServiceError):
    status_code = 409
    public_message = "This thread has no pending interrupt"


class InvalidResumeDecisionError(AgentServiceError):
    status_code = 422
    public_message = "The resume decision does not match the pending interrupt"


class ThreadOwnershipError(AgentServiceError):
    status_code = 403
    public_message = "This thread belongs to another user"


class LegacyThreadError(AgentServiceError):
    status_code = 409
    public_message = (
        "This existing thread has no API user binding; use a new thread_id"
    )


@dataclass(frozen=True)
class InterruptData:
    interrupt_id: str | None
    payload: dict[str, Any]


@dataclass(frozen=True)
class AgentExecutionResult:
    status: Literal["completed", "interrupted"]
    thread_id: str
    answer: str | None = None
    interrupts: list[InterruptData] = field(default_factory=list)


def _config(thread_id: str) -> dict[str, dict[str, str]]:
    return {"configurable": {"thread_id": thread_id}}


def _interrupts_from_result(result: Any) -> tuple[Interrupt, ...]:
    if isinstance(result, dict):
        interrupts = result.get("__interrupt__", ())
    else:
        interrupts = getattr(result, "interrupts", ())
    return tuple(interrupts or ())


def _snapshot_interrupts(snapshot: Any) -> tuple[Interrupt, ...]:
    interrupts = getattr(snapshot, "interrupts", ())
    return tuple(interrupts or ())


def _json_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        # The round trip guarantees that the HTTP response never contains
        # LangGraph/Pydantic objects accidentally returned by a future node.
        return json.loads(json.dumps(value, ensure_ascii=False, default=str))
    return {"message": str(value)}


def _last_answer(result: Any) -> str | None:
    if not isinstance(result, dict):
        return None
    for message in reversed(result.get("messages", [])):
        if isinstance(message, AIMessage) and not message.tool_calls:
            return str(message.content).strip() or None
    final_answer = result.get("final_answer")
    if isinstance(final_answer, str) and final_answer.strip():
        return final_answer.strip()
    return None


def _resume_value(
    decision: Literal["approve", "reject", "retry", "cancel"],
    reason: str | None,
) -> dict[str, object]:
    if decision == "approve":
        return {"approved": True}
    if decision == "reject":
        return {"approved": False, "reason": reason or "用户拒绝"}
    if decision == "retry":
        return {"action": "retry"}
    return {"action": "cancel", "reason": reason or "用户取消重试"}


class AgentService:
    """Synchronous adapter around the compiled graph and shared SQLite handles."""

    def __init__(self, graph: CompiledGraph):
        self.graph = graph
        # SqliteSaver/SqliteStore connections are shared for the lifespan.
        # Serialize graph calls in this minimal version to avoid concurrent use
        # of the same synchronous SQLite connections from worker threads.
        self._invoke_lock = RLock()

    def invoke(
        self,
        *,
        message: str,
        thread_id: str,
        user_id: str,
    ) -> AgentExecutionResult:
        config = _config(thread_id)
        context = AgentContext(user_id=user_id)
        with self._invoke_lock:
            snapshot = self.graph.get_state(config)
            self._verify_thread_owner(snapshot, user_id)
            if _snapshot_interrupts(snapshot):
                raise PendingInterruptError
            result = self.graph.invoke(
                {
                    "messages": [HumanMessage(content=message)],
                    "session_user_id": user_id,
                },
                config=config,
                context=context,
            )
        return self._to_result(result, thread_id)

    def resume(
        self,
        *,
        thread_id: str,
        user_id: str,
        decision: Literal["approve", "reject", "retry", "cancel"],
        reason: str | None = None,
    ) -> AgentExecutionResult:
        config = _config(thread_id)
        context = AgentContext(user_id=user_id)
        with self._invoke_lock:
            snapshot = self.graph.get_state(config)
            self._verify_thread_owner(snapshot, user_id)
            pending_interrupts = _snapshot_interrupts(snapshot)
            if not pending_interrupts:
                raise NoPendingInterruptError
            self._verify_resume_decision(pending_interrupts[0], decision)
            result = self.graph.invoke(
                Command(resume=_resume_value(decision, reason)),
                config=config,
                context=context,
            )
        return self._to_result(result, thread_id)

    @staticmethod
    def _verify_resume_decision(
        pending: Interrupt,
        decision: Literal["approve", "reject", "retry", "cancel"],
    ) -> None:
        payload = pending.value
        is_failure_review = (
            isinstance(payload, dict)
            and payload.get("action") == "tool_failure_review"
        )
        allowed = (
            {"retry", "cancel"}
            if is_failure_review
            else {"approve", "reject"}
        )
        if decision not in allowed:
            raise InvalidResumeDecisionError

    @staticmethod
    def _verify_thread_owner(snapshot: Any, user_id: str) -> None:
        values = getattr(snapshot, "values", {}) or {}
        owner = values.get("session_user_id")
        if owner is not None and owner != user_id:
            raise ThreadOwnershipError
        if values and owner is None:
            # Old CLI checkpoints predate HTTP ownership. Automatically
            # claiming one would expose its conversation to the first caller.
            raise LegacyThreadError

    @staticmethod
    def _to_result(result: Any, thread_id: str) -> AgentExecutionResult:
        interrupts = _interrupts_from_result(result)
        if interrupts:
            return AgentExecutionResult(
                status="interrupted",
                thread_id=thread_id,
                interrupts=[
                    InterruptData(
                        interrupt_id=getattr(item, "id", None),
                        payload=_json_payload(item.value),
                    )
                    for item in interrupts
                ],
            )
        return AgentExecutionResult(
            status="completed",
            thread_id=thread_id,
            answer=_last_answer(result),
        )
