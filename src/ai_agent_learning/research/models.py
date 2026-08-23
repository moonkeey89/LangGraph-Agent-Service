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
