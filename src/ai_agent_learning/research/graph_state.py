from dataclasses import dataclass
from typing import Annotated, Literal, TypeVar

from typing_extensions import TypedDict

from ai_agent_learning.agent.state import ErrorType


ResearchRoute = Literal["knowledge", "analysis", "synthesis", "direct"]
ResearchOutcome = Literal["completed", "blocked", "failed", "needs_review"]
T = TypeVar("T")


def replace_list(_current: list[T], update: list[T]) -> list[T]:
    """Replace list state explicitly so replay never duplicates prior values."""
    return list(update)


class ResearchEvidence(TypedDict):
    knowledge_base_id: str
    document_id: str
    chunk_id: str
    source: str
    page: int | None
    content: str
    score: float


class ResearchSource(TypedDict):
    """Stable source shape compatible with ResearchArtifact.ArtifactSource."""

    knowledge_base_id: str
    document_id: str
    chunk_id: str
    source: str
    page: int | None
    excerpt: str


class ResearchSubagentCall(TypedDict):
    agent_name: str
    status: str


class ResearchState(TypedDict, total=False):
    # Trusted binding copied into checkpoints for later resume validation.
    session_user_id: str
    project_id: str
    task_id: str
    run_id: str
    knowledge_base_id: str | None

    # Immutable task snapshot. Nodes only read these values.
    task_title: str
    task_objective: str
    task_type: str
    acceptance_criteria: Annotated[list[str], replace_list]

    # Execution state. Lists use replacement semantics to avoid replay duplication.
    route: ResearchRoute
    evidence: Annotated[list[ResearchEvidence], replace_list]
    sources: Annotated[list[ResearchSource], replace_list]
    analysis_result: str | None
    draft_answer: str | None
    critic_decision: dict[str, object] | None
    revision_count: int
    max_revisions: int
    subagent_calls: Annotated[list[ResearchSubagentCall], replace_list]

    # Stable public result contract.
    outcome: ResearchOutcome
    final_answer: str
    unresolved_issues: Annotated[list[str], replace_list]
    error: str | None
    error_type: ErrorType | None
    failed_node: str | None


@dataclass(frozen=True)
class ResearchContext:
    """Trusted application binding for one Research Graph invocation."""

    user_id: str
    project_id: str
    task_id: str
    run_id: str
    knowledge_base_id: str | None = None
