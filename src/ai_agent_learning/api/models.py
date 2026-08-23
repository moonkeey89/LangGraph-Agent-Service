from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ai_agent_learning.knowledge.models import validate_knowledge_base_id
from ai_agent_learning.research.service import (
    MAX_PROJECT_DESCRIPTION_LENGTH,
    MAX_PROJECT_NAME_LENGTH,
    MAX_RESEARCH_QUESTION_LENGTH,
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
