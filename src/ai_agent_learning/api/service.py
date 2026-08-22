import json
import logging
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from threading import Event, RLock
from typing import Any, Literal, Protocol

import anyio
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command, Interrupt

from ai_agent_learning.agent.context import AgentContext


logger = logging.getLogger(__name__)


class CompiledGraph(Protocol):
    def invoke(self, input: Any, **kwargs: Any) -> Any: ...

    def stream(self, input: Any, **kwargs: Any) -> Iterator[Any]: ...

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


StreamEventType = Literal[
    "started",
    "progress",
    "token",
    "interrupted",
    "completed",
    "error",
]


@dataclass(frozen=True)
class AgentStreamEvent:
    event: StreamEventType
    data: dict[str, Any]


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

    async def stream(
        self,
        *,
        message: str,
        thread_id: str,
        user_id: str,
    ) -> AsyncIterator[AgentStreamEvent]:
        """Bridge the synchronous graph stream to an async SSE consumer."""
        send_stream, receive_stream = anyio.create_memory_object_stream[
            AgentStreamEvent
        ](max_buffer_size=1)
        stop_requested = Event()

        def produce() -> None:
            try:
                for event in self._stream_sync(
                    message=message,
                    thread_id=thread_id,
                    user_id=user_id,
                ):
                    if stop_requested.is_set():
                        break
                    anyio.from_thread.run(send_stream.send, event)
            finally:
                anyio.from_thread.run(send_stream.aclose)

        async def run_producer() -> None:
            await anyio.to_thread.run_sync(produce, abandon_on_cancel=True)

        try:
            async with anyio.create_task_group() as task_group:
                task_group.start_soon(run_producer)
                async with receive_stream:
                    async for event in receive_stream:
                        yield event
                task_group.cancel_scope.cancel()
        finally:
            # A disconnected client closes this async generator. The producer
            # checks the flag between LangGraph chunks and then closes the
            # synchronous iterator, so it does not continue the remaining run.
            stop_requested.set()

    def _stream_sync(
        self,
        *,
        message: str,
        thread_id: str,
        user_id: str,
    ) -> Iterator[AgentStreamEvent]:
        """Execute Graph exactly once and expose only a safe public projection."""
        yield AgentStreamEvent("started", {"thread_id": thread_id})
        config = _config(thread_id)
        context = AgentContext(user_id=user_id)
        token_buffers: dict[str, list[str]] = {"agent": [], "revise": []}

        try:
            with self._invoke_lock:
                snapshot = self.graph.get_state(config)
                self._verify_thread_owner(snapshot, user_id)
                if _snapshot_interrupts(snapshot):
                    raise PendingInterruptError

                for part in self.graph.stream(
                    {
                        "messages": [HumanMessage(content=message)],
                        "session_user_id": user_id,
                    },
                    config=config,
                    context=context,
                    stream_mode=["updates", "messages"],
                    subgraphs=False,
                    version="v2",
                ):
                    if not isinstance(part, dict):
                        continue
                    mode = part.get("type")
                    data = part.get("data")
                    if mode == "updates":
                        yield from _progress_events(data)
                    elif mode == "messages":
                        _buffer_public_token(data, token_buffers)

                final_snapshot = self.graph.get_state(config)
                interrupts = _snapshot_interrupts(final_snapshot)
                if interrupts:
                    yield AgentStreamEvent(
                        "interrupted",
                        {
                            "thread_id": thread_id,
                            "interrupts": [
                                {
                                    "interrupt_id": getattr(item, "id", None),
                                    "payload": _json_payload(item.value),
                                }
                                for item in interrupts
                            ],
                        },
                    )
                    return

                values = getattr(final_snapshot, "values", {}) or {}
                answer = _last_answer(values)
                for content in _verified_answer_tokens(answer, token_buffers):
                    yield AgentStreamEvent("token", {"content": content})
                yield AgentStreamEvent(
                    "completed",
                    {"thread_id": thread_id, "answer": answer},
                )
        except AgentServiceError as error:
            yield AgentStreamEvent(
                "error",
                {
                    "thread_id": thread_id,
                    "code": _stream_error_code(error),
                    "message": error.public_message,
                },
            )
        except Exception as error:
            logger.exception("LangGraph SSE stream failed", exc_info=error)
            yield AgentStreamEvent(
                "error",
                {
                    "thread_id": thread_id,
                    "code": "internal_error",
                    "message": "Agent stream failed",
                },
            )

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


_SAFE_PROGRESS: dict[str, tuple[str, str]] = {
    "memory_recall": ("memory_recall", "正在检索与当前问题相关的长期记忆"),
    "agent": ("supervisor", "Supervisor 正在分析任务并协调所需能力"),
    "tools": ("tools", "正在执行 Agent 选择的能力"),
    "tool_success": ("tools", "工具执行完成，正在返回 Agent"),
    "human_review": ("human_review", "工具失败，正在准备人工复核"),
    "failure": ("failure", "正在生成安全的失败降级回答"),
    "capture_draft": ("draft", "Supervisor 回答草稿已生成"),
    "critic": ("critic", "Critic 正在审查回答完整性"),
    "revise": ("revision", "正在根据审查结果修订回答"),
    "finalize": ("finalize", "最终回答已经确定"),
    "memory_manager": ("memory_manager", "正在评估长期记忆决策"),
    "memory_executor": ("memory_executor", "长期记忆决策处理完成"),
}


def _progress_events(data: Any) -> Iterator[AgentStreamEvent]:
    if not isinstance(data, dict):
        return
    for node_name in data:
        progress = _SAFE_PROGRESS.get(str(node_name))
        if progress is None:
            continue
        stage, description = progress
        yield AgentStreamEvent(
            "progress",
            {
                "stage": stage,
                "node": str(node_name),
                "message": description,
            },
        )


def _buffer_public_token(
    data: Any,
    token_buffers: dict[str, list[str]],
) -> None:
    if not isinstance(data, (tuple, list)) or len(data) != 2:
        return
    message_chunk, metadata = data
    if not isinstance(metadata, dict):
        return
    node_name = str(metadata.get("langgraph_node", ""))
    if node_name not in token_buffers:
        return
    content = _safe_text_content(getattr(message_chunk, "content", None))
    if content:
        token_buffers[node_name].append(content)


def _safe_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    text_parts: list[str] = []
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "text":
            continue
        text = block.get("text")
        if isinstance(text, str):
            text_parts.append(text)
    return "".join(text_parts)


def _verified_answer_tokens(
    answer: str | None,
    token_buffers: dict[str, list[str]],
) -> list[str]:
    if not answer:
        return []
    # Revision is the final LLM call when Critic requested a change. Otherwise
    # the Supervisor draft becomes final. Never expose Critic/Memory/Subagent
    # model chunks, and only release a buffer that equals the finalized answer.
    for node_name in ("revise", "agent"):
        fragments = token_buffers[node_name]
        combined = "".join(fragments)
        if combined == answer or combined.strip() == answer.strip():
            return fragments
    return []


def _stream_error_code(error: AgentServiceError) -> str:
    if isinstance(error, ThreadOwnershipError):
        return "thread_forbidden"
    if isinstance(error, LegacyThreadError):
        return "legacy_thread_conflict"
    if isinstance(error, PendingInterruptError):
        return "pending_interrupt"
    return "request_rejected"
