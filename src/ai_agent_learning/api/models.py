from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ai_agent_learning.knowledge.models import validate_knowledge_base_id
from ai_agent_learning.research.service import (
    MAX_ACCEPTANCE_CRITERIA,
    MAX_ACCEPTANCE_CRITERION_LENGTH,
    MAX_ARTIFACT_CONTENT_LENGTH,
    MAX_ARTIFACT_SOURCES,
    MAX_ARTIFACT_TITLE_LENGTH,
    MAX_PROJECT_DESCRIPTION_LENGTH,
    MAX_PROJECT_NAME_LENGTH,
    MAX_RESEARCH_QUESTION_LENGTH,
    MAX_TASK_OBJECTIVE_LENGTH,
    MAX_TASK_TITLE_LENGTH,
)
from ai_agent_learning.research.task_state import (
    MAX_TASK_RESULT_SUMMARY_LENGTH,
    MAX_TASK_TRANSITION_REASON_LENGTH,
)


MAX_IDENTIFIER_LENGTH = 128


def _normalized_identifier(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} 不能为空")
    if len(normalized) > MAX_IDENTIFIER_LENGTH:
        raise ValueError(
            f"{field_name} 不能超过 {MAX_IDENTIFIER_LENGTH} 个字符"
        )
    return normalized


class InvokeRequest(BaseModel):
    message: str = Field(min_length=1, max_length=20_000)
    thread_id: str = Field(min_length=1, max_length=MAX_IDENTIFIER_LENGTH)
    knowledge_base_id: str | None = None

    @field_validator("message")
    @classmethod
    def normalize_message(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("message 不能为空")
        return normalized

    @field_validator("thread_id")
    @classmethod
    def normalize_thread_id(cls, value: str) -> str:
        return _normalized_identifier(value, "thread_id")

    @field_validator("knowledge_base_id")
    @classmethod
    def normalize_knowledge_base_id(cls, value: str | None) -> str | None:
        return None if value is None else validate_knowledge_base_id(value)


class ResumeRequest(BaseModel):
    thread_id: str = Field(min_length=1, max_length=MAX_IDENTIFIER_LENGTH)
    decision: Literal["approve", "reject", "retry", "cancel"]
    reason: str | None = Field(default=None, max_length=500)
    knowledge_base_id: str | None = None

    @field_validator("thread_id")
    @classmethod
    def normalize_thread_id(cls, value: str) -> str:
        return _normalized_identifier(value, "thread_id")

    @field_validator("knowledge_base_id")
    @classmethod
    def normalize_knowledge_base_id(cls, value: str | None) -> str | None:
        return None if value is None else validate_knowledge_base_id(value)


class InterruptResponse(BaseModel):
    interrupt_id: str | None = None
    payload: dict[str, Any]


class KnowledgeSourceResponse(BaseModel):
    source: str
    page: int | None = None
    document_id: str
    chunk_id: str
    score: float


class AgentResponse(BaseModel):
    status: Literal["completed", "interrupted"]
    thread_id: str
    answer: str | None = None
    interrupts: list[InterruptResponse] = Field(default_factory=list)
    sources: list[KnowledgeSourceResponse] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class CreateKnowledgeBaseRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    description: str = Field(default="", max_length=1000)


class KnowledgeBaseResponse(BaseModel):
    knowledge_base_id: str
    owner_user_id: str
    name: str
    description: str
    created_at: str
    updated_at: str


class KnowledgeDocumentResponse(BaseModel):
    document_id: str
    knowledge_base_id: str
    original_filename: str
    content_hash: str
    content_type: str
    size: int
    status: Literal["processing", "ready", "failed"]
    chunk_count: int
    error_message: str | None
    created_at: str
    updated_at: str


class KnowledgeUploadItemResponse(BaseModel):
    document: KnowledgeDocumentResponse
    duplicate: bool


class KnowledgeUploadResponse(BaseModel):
    items: list[KnowledgeUploadItemResponse]


class CreateResearchProjectRequest(BaseModel):
    name: str = Field(min_length=1, max_length=MAX_PROJECT_NAME_LENGTH)
    description: str = Field(
        default="",
        max_length=MAX_PROJECT_DESCRIPTION_LENGTH,
    )
    research_question: str = Field(
        default="",
        max_length=MAX_RESEARCH_QUESTION_LENGTH,
    )
    status: Literal["draft", "active", "archived"] = "draft"
    default_knowledge_base_id: str | None = None

    model_config = ConfigDict(extra="forbid")

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("name 不能为空")
        return normalized

    @field_validator("description", "research_question")
    @classmethod
    def normalize_optional_text(cls, value: str) -> str:
        return value.strip()

    @field_validator("default_knowledge_base_id")
    @classmethod
    def normalize_project_knowledge_base_id(
        cls,
        value: str | None,
    ) -> str | None:
        return None if value is None else validate_knowledge_base_id(value)


class UpdateResearchProjectRequest(BaseModel):
    name: str | None = Field(default=None, max_length=MAX_PROJECT_NAME_LENGTH)
    description: str | None = Field(
        default=None,
        max_length=MAX_PROJECT_DESCRIPTION_LENGTH,
    )
    research_question: str | None = Field(
        default=None,
        max_length=MAX_RESEARCH_QUESTION_LENGTH,
    )
    status: Literal["draft", "active", "archived"] | None = None
    default_knowledge_base_id: str | None = None

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_partial_update(self) -> "UpdateResearchProjectRequest":
        non_nullable = {
            "name",
            "description",
            "research_question",
            "status",
        }
        for field_name in non_nullable & self.model_fields_set:
            if getattr(self, field_name) is None:
                raise ValueError(f"{field_name} 不能为 null")
        if "name" in self.model_fields_set and not (self.name or "").strip():
            raise ValueError("name 不能为空")
        return self

    @field_validator("name", "description", "research_question")
    @classmethod
    def normalize_update_text(cls, value: str | None) -> str | None:
        return None if value is None else value.strip()

    @field_validator("default_knowledge_base_id")
    @classmethod
    def normalize_update_knowledge_base_id(
        cls,
        value: str | None,
    ) -> str | None:
        return None if value is None else validate_knowledge_base_id(value)

    def changes(self) -> dict[str, Any]:
        return self.model_dump(exclude_unset=True)


class ResearchProjectResponse(BaseModel):
    project_id: str
    owner_user_id: str
    name: str
    description: str
    research_question: str
    status: Literal["draft", "active", "archived"]
    default_knowledge_base_id: str | None
    created_at: str
    updated_at: str

    model_config = ConfigDict(from_attributes=True)


class ErrorResponse(BaseModel):
    detail: str


TaskType = Literal[
    "literature_review",
    "analysis",
    "synthesis",
    "general",
]
TaskStatus = Literal[
    "pending",
    "running",
    "blocked",
    "completed",
    "failed",
    "cancelled",
]


class CreateResearchTaskRequest(BaseModel):
    title: str = Field(min_length=1, max_length=MAX_TASK_TITLE_LENGTH)
    objective: str = Field(default="", max_length=MAX_TASK_OBJECTIVE_LENGTH)
    task_type: TaskType = "general"
    acceptance_criteria: list[str] = Field(
        default_factory=list,
        max_length=MAX_ACCEPTANCE_CRITERIA,
    )

    model_config = ConfigDict(extra="forbid")

    @field_validator("title")
    @classmethod
    def normalize_task_title(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("title 不能为空")
        return normalized

    @field_validator("objective")
    @classmethod
    def normalize_task_objective(cls, value: str) -> str:
        return value.strip()

    @field_validator("acceptance_criteria")
    @classmethod
    def normalize_acceptance_criteria(cls, values: list[str]) -> list[str]:
        normalized: list[str] = []
        for value in values:
            criterion = value.strip()
            if not criterion:
                raise ValueError("验收标准不能为空")
            if len(criterion) > MAX_ACCEPTANCE_CRITERION_LENGTH:
                raise ValueError(
                    "单条验收标准不能超过 "
                    f"{MAX_ACCEPTANCE_CRITERION_LENGTH} 个字符"
                )
            normalized.append(criterion)
        return normalized


class UpdateResearchTaskRequest(BaseModel):
    title: str | None = Field(default=None, max_length=MAX_TASK_TITLE_LENGTH)
    objective: str | None = Field(
        default=None,
        max_length=MAX_TASK_OBJECTIVE_LENGTH,
    )
    task_type: TaskType | None = None
    acceptance_criteria: list[str] | None = Field(
        default=None,
        max_length=MAX_ACCEPTANCE_CRITERIA,
    )

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_task_partial_update(self) -> "UpdateResearchTaskRequest":
        for field_name in self.model_fields_set:
            if getattr(self, field_name) is None:
                raise ValueError(f"{field_name} 不能为 null")
        if "title" in self.model_fields_set and not (self.title or "").strip():
            raise ValueError("title 不能为空")
        return self

    @field_validator("title", "objective")
    @classmethod
    def normalize_task_update_text(cls, value: str | None) -> str | None:
        return None if value is None else value.strip()

    @field_validator("acceptance_criteria")
    @classmethod
    def normalize_task_update_criteria(
        cls,
        values: list[str] | None,
    ) -> list[str] | None:
        if values is None:
            return None
        return CreateResearchTaskRequest.normalize_acceptance_criteria(values)

    def changes(self) -> dict[str, Any]:
        return self.model_dump(exclude_unset=True)


class TransitionResearchTaskRequest(BaseModel):
    target_status: TaskStatus
    reason: str | None = Field(
        default=None,
        max_length=MAX_TASK_TRANSITION_REASON_LENGTH,
    )
    result_summary: str | None = Field(
        default=None,
        max_length=MAX_TASK_RESULT_SUMMARY_LENGTH,
    )

    model_config = ConfigDict(extra="forbid")

    @field_validator("reason", "result_summary")
    @classmethod
    def normalize_transition_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None


class ResearchTaskResponse(BaseModel):
    task_id: str
    project_id: str
    title: str
    objective: str
    task_type: TaskType
    status: TaskStatus
    acceptance_criteria: list[str]
    result_summary: str | None
    error_message: str | None
    created_at: str
    updated_at: str
    started_at: str | None
    completed_at: str | None

    model_config = ConfigDict(from_attributes=True)


ArtifactType = Literal["note", "literature_review", "analysis", "report"]
ArtifactStatus = Literal["draft", "final"]
ArtifactCreator = Literal["user", "agent"]


class CreateResearchArtifactRequest(BaseModel):
    title: str = Field(min_length=1, max_length=MAX_ARTIFACT_TITLE_LENGTH)
    content: str = Field(min_length=1, max_length=MAX_ARTIFACT_CONTENT_LENGTH)
    artifact_type: ArtifactType = "note"
    task_id: str | None = Field(default=None, max_length=MAX_IDENTIFIER_LENGTH)
    source_chunk_ids: list[str] = Field(
        default_factory=list,
        max_length=MAX_ARTIFACT_SOURCES,
    )

    model_config = ConfigDict(extra="forbid")

    @field_validator("title", "content")
    @classmethod
    def normalize_artifact_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("字段不能为空")
        return normalized

    @field_validator("task_id")
    @classmethod
    def normalize_optional_task_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _normalized_identifier(value, "task_id")

    @field_validator("source_chunk_ids")
    @classmethod
    def normalize_source_chunk_ids(cls, values: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for value in values:
            chunk_id = _normalized_identifier(value, "source_chunk_id")
            if chunk_id not in seen:
                normalized.append(chunk_id)
                seen.add(chunk_id)
        return normalized


class UpdateResearchArtifactRequest(BaseModel):
    title: str | None = Field(default=None, max_length=MAX_ARTIFACT_TITLE_LENGTH)
    content: str | None = Field(
        default=None,
        max_length=MAX_ARTIFACT_CONTENT_LENGTH,
    )
    artifact_type: ArtifactType | None = None
    source_chunk_ids: list[str] | None = Field(
        default=None,
        max_length=MAX_ARTIFACT_SOURCES,
    )

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_artifact_partial_update(self) -> "UpdateResearchArtifactRequest":
        for field_name in self.model_fields_set:
            if getattr(self, field_name) is None:
                raise ValueError(f"{field_name} 不能为 null")
        return self

    @field_validator("title", "content")
    @classmethod
    def normalize_artifact_update_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("字段不能为空")
        return normalized

    @field_validator("source_chunk_ids")
    @classmethod
    def normalize_artifact_update_sources(
        cls,
        values: list[str] | None,
    ) -> list[str] | None:
        if values is None:
            return None
        return CreateResearchArtifactRequest.normalize_source_chunk_ids(values)

    def changes(self) -> dict[str, Any]:
        return self.model_dump(exclude_unset=True)


class ArtifactSourceResponse(BaseModel):
    knowledge_base_id: str
    document_id: str
    chunk_id: str
    source: str
    page: int | None
    excerpt: str

    model_config = ConfigDict(from_attributes=True)


class ResearchArtifactResponse(BaseModel):
    artifact_id: str
    project_id: str
    task_id: str | None
    title: str
    artifact_type: ArtifactType
    content: str
    status: ArtifactStatus
    created_by: ArtifactCreator
    sources: list[ArtifactSourceResponse]
    origin_run_id: str | None
    created_at: str
    updated_at: str
    finalized_at: str | None

    model_config = ConfigDict(from_attributes=True)


AgentRunStatusValue = Literal[
    "pending",
    "running",
    "interrupted",
    "completed",
    "failed",
    "cancelled",
]


class AgentRunResponse(BaseModel):
    run_id: str
    task_id: str
    thread_id: str
    attempt_number: int
    status: AgentRunStatusValue
    outcome: Literal["completed", "blocked", "failed", "needs_review"] | None
    output_artifact_id: str | None
    error_message: str | None
    created_at: str
    started_at: str | None
    finished_at: str | None
    updated_at: str

    model_config = ConfigDict(from_attributes=True)


class ResearchExecutionResponse(BaseModel):
    run_id: str
    task_id: str
    status: AgentRunStatusValue
    outcome: Literal["completed", "blocked", "failed", "needs_review"]
    output_artifact_id: str | None
    error: str | None

    model_config = ConfigDict(from_attributes=True)
