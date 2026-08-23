import re
from dataclasses import asdict, dataclass
from typing import Any, Literal


KNOWLEDGE_BASE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


def validate_knowledge_base_id(value: str) -> str:
    normalized = value.strip()
    if not KNOWLEDGE_BASE_ID_PATTERN.fullmatch(normalized):
        raise ValueError(
            "knowledge_base_id 只能包含字母、数字、下划线和短横线，"
            "必须以字母或数字开头且长度不超过64"
        )
    return normalized


@dataclass(frozen=True)
class KnowledgeChunk:
    content: str
    knowledge_base_id: str
    document_id: str
    source: str
    page: int | None
    chunk_id: str

    def metadata(self) -> dict[str, str | int]:
        return {
            "knowledge_base_id": self.knowledge_base_id,
            "document_id": self.document_id,
            "source": self.source,
            "page": self.page if self.page is not None else -1,
            "chunk_id": self.chunk_id,
        }


@dataclass(frozen=True)
class KnowledgeSearchResult:
    content: str
    score: float
    source: str
    page: int | None
    document_id: str
    chunk_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def source_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "page": self.page,
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "score": self.score,
        }


@dataclass(frozen=True)
class KnowledgeSearchResponse:
    status: Literal["found", "no_evidence"]
    knowledge_base_id: str
    results: list[KnowledgeSearchResult]
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "knowledge_base_id": self.knowledge_base_id,
            "results": [item.to_dict() for item in self.results],
            "message": self.message,
        }


@dataclass(frozen=True)
class IngestionResult:
    knowledge_base_id: str
    document_id: str
    source: str
    chunk_count: int
    replaced_chunk_count: int

