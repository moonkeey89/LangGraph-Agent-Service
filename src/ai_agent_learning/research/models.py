from dataclasses import dataclass
from typing import Literal


ResearchProjectStatus = Literal["draft", "active", "archived"]
RESEARCH_PROJECT_STATUSES = frozenset({"draft", "active", "archived"})


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
