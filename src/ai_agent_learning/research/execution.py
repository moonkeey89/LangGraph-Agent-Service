import logging
from dataclasses import dataclass
from threading import RLock
from typing import Any, Protocol

from ai_agent_learning.research.graph_state import (
    ResearchContext,
    ResearchOutcome,
    ResearchState,
)
from ai_agent_learning.research.run_state import clean_run_error
from ai_agent_learning.research.service import (
    ResearchService,
    ResearchServiceError,
)


logger = logging.getLogger(__name__)
RESEARCH_CHECKPOINT_NAMESPACE = "researchflow"


class ResearchCompiledGraph(Protocol):
    def invoke(
        self,
        input: ResearchState,
        *,
        config: dict[str, Any],
        context: ResearchContext,
    ) -> dict[str, object]: ...


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


class ResearchExecutionService:
    """Coordinate domain persistence around one non-streaming Graph invocation."""

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
        project, task, run = self.research_service.start_execution(
            project_id,
            task_id,
            user_id,
        )
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

        try:
            with self._invoke_lock:
                graph_result = self.graph.invoke(
                    state,
                    config=config,
                    context=context,
                )
            outcome = self._outcome(graph_result)
            return self._finish(
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
                return self._finish(
                    project_id=project.project_id,
                    task_id=task.task_id,
                    run_id=run.run_id,
                    user_id=user_id,
                    outcome="failed",
                    graph_result={"error": safe_error},
                )
            except Exception as finish_error:
                logger.exception(
                    "Research execution failure finalization also failed for run_id=%s",
                    run.run_id,
                    exc_info=finish_error,
                )
                raise ResearchExecutionError from finish_error

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
