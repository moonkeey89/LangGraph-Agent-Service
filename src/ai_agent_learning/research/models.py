from dataclasses import dataclass
from typing import Literal


ResearchProjectStatus = Literal["draft", "active", "archived"]
RESEARCH_PROJECT_STATUSES = frozenset({"draft", "active", "archived"})
ResearchTaskType = Literal[
    "literature_review",
    "analysis",
    "synthesis",
    "general",
]
RESEARCH_TASK_TYPES = frozenset(
    {"literature_review", "analysis", "synthesis", "general"}
)
ResearchTaskStatus = Literal[
    "pending",
    "running",
    "blocked",
    "completed",
    "failed",
    "cancelled",
]
RESEARCH_TASK_STATUSES = frozenset(
    {"pending", "running", "blocked", "completed", "failed", "cancelled"}
)
ResearchArtifactType = Literal[
    "note",
    "literature_review",
    "analysis",
    "report",
]
RESEARCH_ARTIFACT_TYPES = frozenset(
    {"note", "literature_review", "analysis", "report"}
)
ResearchArtifactStatus = Literal["draft", "final"]
RESEARCH_ARTIFACT_STATUSES = frozenset({"draft", "final"})
ResearchArtifactCreator = Literal["user", "agent"]
RESEARCH_ARTIFACT_CREATORS = frozenset({"user", "agent"})
AgentRunStatus = Literal[
    "pending",
    "running",
    "interrupted",
    "completed",
    "failed",
    "cancelled",
]
AGENT_RUN_STATUSES = frozenset(
    {"pending", "running", "interrupted", "completed", "failed", "cancelled"}
)
AgentRunOutcome = Literal["completed", "blocked", "failed", "needs_review"]
AGENT_RUN_OUTCOMES = frozenset(
    {"completed", "blocked", "failed", "needs_review"}
)


@dataclass(frozen=True)
class ResearchProject:
    """Persisted ResearchFlow project owned by one application user."""

    project_id: str
    owner_user_id: str
    name: str
    description: str
    research_question: str
    status: ResearchProjectStatus
    default_knowledge_base_id: str | None
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class ResearchTask:
    """One user-owned-through-project unit of research work."""

    task_id: str
    project_id: str
    title: str
    objective: str
    task_type: ResearchTaskType
    status: ResearchTaskStatus
    acceptance_criteria: list[str]
    result_summary: str | None
    error_message: str | None
    created_at: str
    updated_at: str
    started_at: str | None
    completed_at: str | None


@dataclass(frozen=True)
class ArtifactSource:
    """Immutable evidence snapshot resolved from one trusted RAG chunk."""

    knowledge_base_id: str
    document_id: str
    chunk_id: str
    source: str
    page: int | None
    excerpt: str


@dataclass(frozen=True)
class ResearchArtifact:
    """A draft or finalized research output owned through its project."""

    artifact_id: str
    project_id: str
    task_id: str | None
    title: str
    artifact_type: ResearchArtifactType
    content: str
    status: ResearchArtifactStatus
    created_by: ResearchArtifactCreator
    sources: list[ArtifactSource]
    created_at: str
    updated_at: str
    finalized_at: str | None
    origin_run_id: str | None = None


@dataclass(frozen=True)
class AgentRun:
    """Durable metadata for one execution attempt of a research task."""

    run_id: str
    task_id: str
    thread_id: str
    attempt_number: int
    status: AgentRunStatus
    outcome: AgentRunOutcome | None
    output_artifact_id: str | None
    error_message: str | None
    created_at: str
    started_at: str | None
    finished_at: str | None
    updated_at: str

    @property
    def final_artifact_id(self) -> str | None:
        """Deprecated read-only alias for records created before schema v5."""
        return self.output_artifact_id
