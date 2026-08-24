import logging
from dataclasses import dataclass, field
from queue import Queue
from threading import Event, RLock, Thread
from typing import Any, AsyncIterator, Iterator, Literal, Protocol

import anyio

from ai_agent_learning.research.graph_state import (
    ResearchContext,
    ResearchOutcome,
    ResearchState,
)
from ai_agent_learning.research.run_state import clean_run_error
from ai_agent_learning.research.service import ResearchService, ResearchServiceError


logger = logging.getLogger(__name__)
RESEARCH_CHECKPOINT_NAMESPACE = "researchflow"
ResearchStreamEventType = Literal[
    "run_started",
    "task_status",
    "agent_progress",
    "evidence_found",
    "token",
    "artifact_created",
    "run_completed",
    "run_blocked",
    "run_needs_review",
    "run_failed",
]
RESEARCH_TERMINAL_EVENTS = frozenset(
    {"run_completed", "run_blocked", "run_needs_review", "run_failed"}
)
_SAFE_PROGRESS: dict[str, tuple[str, str]] = {
    "research_validate_binding": ("validate_binding", "正在验证任务上下文"),
    "research_supervisor": ("planning", "正在规划科研任务"),
    "research_evidence_agent": ("evidence", "正在检索研究证据"),
    "research_analysis_agent": ("analysis", "正在分析数据与指标"),
    "research_synthesize": ("synthesize", "正在生成候选成果"),
    "research_critic": ("critic", "正在检查证据和验收条件"),
    "research_revise": ("revise", "正在修订成果"),
    "research_finalize": ("finalize", "正在整理最终结果"),
}
_PUBLIC_TOKEN_NODES = ("research_revise", "research_synthesize")
_QUEUE_END = object()


class ResearchCompiledGraph(Protocol):
    def stream(
        self,
        input: ResearchState,
        *,
        config: dict[str, Any],
        context: ResearchContext,
        stream_mode: list[str],
        subgraphs: bool,
        version: str,
    ) -> Iterator[dict[str, object]]: ...


class ResearchExecutionError(ResearchServiceError):
    status_code = 500
    public_message = "Research execution could not be completed"


@dataclass(frozen=True)
class ResearchExecutionResult:
    run_id: str
    task_id: str
    status: str
    outcome: ResearchOutcome
    output_artifact_id: str | None
    error: str | None


@dataclass(frozen=True)
class ResearchStreamEvent:
    event: ResearchStreamEventType
    data: dict[str, object]
    result: ResearchExecutionResult | None = field(
        default=None,
        repr=False,
        compare=False,
    )


class ResearchExecutionService:
    """Coordinate one Graph execution for both non-streaming and SSE callers."""

    def __init__(
        self,
        research_service: ResearchService,
        graph: ResearchCompiledGraph,
    ):
        self.research_service = research_service
        self.graph = graph
        self._invoke_lock = RLock()

    def execute_task(
        self,
        *,
        project_id: str,
        task_id: str,
        user_id: str,
    ) -> ResearchExecutionResult:
        """Consume the same domain event stream used by SSE without re-running it."""
        result: ResearchExecutionResult | None = None
        for event in self._execute_sync(
            project_id=project_id,
            task_id=task_id,
            user_id=user_id,
        ):
            if event.result is not None:
                result = event.result
        if result is None:
            raise ResearchExecutionError
        return result

    async def execute_stream(
        self,
        *,
        project_id: str,
        task_id: str,
        user_id: str,
    ) -> AsyncIterator[ResearchStreamEvent]:
        """Keep business execution alive when the HTTP consumer disconnects."""
        event_queue: Queue[ResearchStreamEvent | object] = Queue()
        consumer_closed = Event()

        def produce() -> None:
            try:
                for event in self._execute_sync(
                    project_id=project_id,
                    task_id=task_id,
                    user_id=user_id,
                ):
                    if not consumer_closed.is_set():
                        event_queue.put(event)
            except ResearchServiceError as error:
                logger.info(
                    "Research stream rejected task_id=%s error=%s",
                    task_id,
                    type(error).__name__,
                )
                if not consumer_closed.is_set():
                    event_queue.put(
                        ResearchStreamEvent(
                            "run_failed",
                            {
                                "task_id": task_id,
                                "code": "request_rejected",
                                "message": error.public_message,
                            },
                        )
                    )
            except Exception as error:
                logger.exception(
                    "Research SSE producer failed task_id=%s",
                    task_id,
                    exc_info=error,
                )
                if not consumer_closed.is_set():
                    event_queue.put(
                        ResearchStreamEvent(
                            "run_failed",
                            {
                                "task_id": task_id,
                                "code": "internal_error",
                                "message": "Research execution failed",
                            },
                        )
                    )
            finally:
                if not consumer_closed.is_set():
                    event_queue.put(_QUEUE_END)

        Thread(
            target=produce,
            name=f"research-sse-{task_id[:24]}",
            daemon=True,
        ).start()
        try:
            while True:
                item = await anyio.to_thread.run_sync(
                    event_queue.get,
                    abandon_on_cancel=True,
                )
                if item is _QUEUE_END:
                    return
                assert isinstance(item, ResearchStreamEvent)
                yield item
                if item.event in RESEARCH_TERMINAL_EVENTS:
                    return
        finally:
            # This only stops event delivery. The producer deliberately keeps
            # running through transaction B so the durable Run remains explainable.
            consumer_closed.set()

    def _execute_sync(
        self,
        *,
        project_id: str,
        task_id: str,
        user_id: str,
    ) -> Iterator[ResearchStreamEvent]:
        project, task, run = self.research_service.start_execution(
            project_id,
            task_id,
            user_id,
        )
        yield ResearchStreamEvent(
            "run_started",
            {
                "run_id": run.run_id,
                "task_id": task.task_id,
                "thread_id": run.thread_id,
            },
        )
        yield ResearchStreamEvent(
            "task_status",
            {"run_id": run.run_id, "task_id": task.task_id, "status": "running"},
        )
        state, context, config = self._graph_binding(
            project=project,
            task=task,
            run=run,
            user_id=user_id,
        )
        graph_result: dict[str, object] = {}
        token_buffers: dict[str, list[str]] = {
            node_name: [] for node_name in _PUBLIC_TOKEN_NODES
        }
        progress_seen: set[str] = set()
        evidence_emitted = False

        try:
            with self._invoke_lock:
                for part in self.graph.stream(
                    state,
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
                    if mode == "updates" and isinstance(data, dict):
                        for node_name, node_update in data.items():
                            normalized_node = str(node_name)
                            if (
                                normalized_node in _SAFE_PROGRESS
                                and normalized_node not in progress_seen
                            ):
                                stage, message = _SAFE_PROGRESS[normalized_node]
                                progress_seen.add(normalized_node)
                                yield ResearchStreamEvent(
                                    "agent_progress",
                                    {"stage": stage, "message": message},
                                )
                            if isinstance(node_update, dict):
                                graph_result.update(node_update)
                            if (
                                normalized_node == "research_evidence_agent"
                                and not evidence_emitted
                            ):
                                evidence_event = _evidence_event(node_update)
                                if evidence_event is not None:
                                    evidence_emitted = True
                                    yield evidence_event
                    elif mode == "messages":
                        _buffer_research_token(data, token_buffers)

            outcome = self._outcome(graph_result)
            result = self._finish(
                project_id=project.project_id,
                task_id=task.task_id,
                run_id=run.run_id,
                user_id=user_id,
                outcome=outcome,
                graph_result=graph_result,
            )
        except Exception as error:
            logger.exception(
                "Research Graph execution failed for run_id=%s",
                run.run_id,
            )
            safe_error = clean_run_error(str(error)) or "科研任务执行失败"
            try:
                graph_result = {"error": safe_error}
                result = self._finish(
                    project_id=project.project_id,
                    task_id=task.task_id,
                    run_id=run.run_id,
                    user_id=user_id,
                    outcome="failed",
                    graph_result=graph_result,
                )
            except Exception as finish_error:
                logger.exception(
                    "Research execution failure finalization also failed for run_id=%s",
                    run.run_id,
                    exc_info=finish_error,
                )
                raise ResearchExecutionError from finish_error

        answer = str(graph_result.get("final_answer") or "")
        for content in _verified_research_tokens(answer, token_buffers):
            yield ResearchStreamEvent("token", {"content": content})
        if result.output_artifact_id is not None:
            yield ResearchStreamEvent(
                "artifact_created",
                {
                    "artifact_id": result.output_artifact_id,
                    "status": "draft",
                    "created_by": "agent",
                },
            )
        yield _terminal_event(result, answer=answer)

    @staticmethod
    def _graph_binding(*, project, task, run, user_id: str):
        state: ResearchState = {
            "session_user_id": user_id,
            "project_id": project.project_id,
            "task_id": task.task_id,
            "run_id": run.run_id,
            "knowledge_base_id": project.default_knowledge_base_id,
            "task_title": task.title,
            "task_objective": task.objective,
            "task_type": task.task_type,
            "acceptance_criteria": list(task.acceptance_criteria),
        }
        context = ResearchContext(
            user_id=user_id,
            project_id=project.project_id,
            task_id=task.task_id,
            run_id=run.run_id,
            knowledge_base_id=project.default_knowledge_base_id,
        )
        config = {
            "configurable": {
                "thread_id": run.thread_id,
                "checkpoint_ns": RESEARCH_CHECKPOINT_NAMESPACE,
            }
        }
        return state, context, config

    def _finish(
        self,
        *,
        project_id: str,
        task_id: str,
        run_id: str,
        user_id: str,
        outcome: ResearchOutcome,
        graph_result: dict[str, object],
    ) -> ResearchExecutionResult:
        raw_sources = graph_result.get("sources", [])
        sources = raw_sources if isinstance(raw_sources, list) else []
        raw_issues = graph_result.get("unresolved_issues", [])
        issues = raw_issues if isinstance(raw_issues, list) else []
        task, run, artifact = self.research_service.finish_execution(
            project_id,
            task_id,
            run_id,
            user_id,
            outcome=outcome,
            final_answer=str(graph_result.get("final_answer") or ""),
            sources=sources,
            unresolved_issues=[str(item) for item in issues],
            error=(
                str(graph_result["error"])
                if graph_result.get("error") is not None
                else None
            ),
        )
        return ResearchExecutionResult(
            run_id=run.run_id,
            task_id=task.task_id,
            status=run.status,
            outcome=outcome,
            output_artifact_id=(
                artifact.artifact_id if artifact is not None else None
            ),
            error=run.error_message,
        )

    @staticmethod
    def _outcome(result: object) -> ResearchOutcome:
        if not isinstance(result, dict):
            raise ValueError("Research Graph returned an invalid result")
        outcome = result.get("outcome")
        if outcome not in {"completed", "blocked", "failed", "needs_review"}:
            raise ValueError("Research Graph returned an invalid outcome")
        return outcome


def _evidence_event(node_update: object) -> ResearchStreamEvent | None:
    if not isinstance(node_update, dict):
        return None
    raw_sources = node_update.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        return None
    summaries: list[dict[str, object]] = []
    for item in raw_sources[:10]:
        if not isinstance(item, dict):
            continue
        source = item.get("source")
        page = item.get("page")
        if not isinstance(source, str) or not source.strip():
            continue
        summaries.append(
            {
                "source": " ".join(source.strip().splitlines())[:255],
                "page": page if isinstance(page, int) else None,
            }
        )
    if not summaries:
        return None
    return ResearchStreamEvent(
        "evidence_found",
        {"count": len(raw_sources), "sources": summaries},
    )


def _buffer_research_token(
    data: object,
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


def _safe_text_content(content: object) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict) or item.get("type") != "text":
            continue
        text = item.get("text")
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts)


def _verified_research_tokens(
    answer: str,
    token_buffers: dict[str, list[str]],
) -> list[str]:
    if not answer:
        return []
    for node_name in _PUBLIC_TOKEN_NODES:
        fragments = token_buffers[node_name]
        combined = "".join(fragments)
        if combined == answer or combined.strip() == answer.strip():
            return fragments
    return []


def _terminal_event(
    result: ResearchExecutionResult,
    *,
    answer: str,
) -> ResearchStreamEvent:
    event_by_outcome: dict[ResearchOutcome, ResearchStreamEventType] = {
        "completed": "run_completed",
        "blocked": "run_blocked",
        "needs_review": "run_needs_review",
        "failed": "run_failed",
    }
    data: dict[str, object] = {
        "run_id": result.run_id,
        "task_id": result.task_id,
        "status": result.status,
        "outcome": result.outcome,
        "output_artifact_id": result.output_artifact_id,
        "error": result.error,
    }
    if result.outcome in {"completed", "needs_review"}:
        data["answer"] = answer
    if result.outcome == "failed":
        data["code"] = "research_execution_failed"
        data["message"] = result.error or "Research execution failed"
    return ResearchStreamEvent(
        event_by_outcome[result.outcome],
        data,
        result=result,
    )
