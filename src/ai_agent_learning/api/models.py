from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from ai_agent_learning.knowledge.models import validate_knowledge_base_id


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
