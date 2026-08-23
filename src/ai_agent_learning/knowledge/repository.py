from pathlib import Path
from collections.abc import Callable
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection
from langchain_core.embeddings import Embeddings

from ai_agent_learning.knowledge.models import (
    KnowledgeChunk,
    KnowledgeSearchResponse,
    KnowledgeSearchResult,
    validate_knowledge_base_id,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
KNOWLEDGE_CHROMA_PATH = PROJECT_ROOT / "data" / "knowledge_chroma"
KNOWLEDGE_COLLECTION_NAME = "knowledge_chunks"


def resolve_knowledge_directory(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


class ChromaKnowledgeRepository:
    """Persistent RAG storage, deliberately separate from user memory Store."""

    def __init__(
        self,
        *,
        persist_directory: Path,
        embeddings: Embeddings,
        collection_name: str = KNOWLEDGE_COLLECTION_NAME,
    ):
        persist_directory.mkdir(parents=True, exist_ok=True)
        self.persist_directory = persist_directory
        self.embeddings = embeddings
        self.client = chromadb.PersistentClient(path=str(persist_directory))
        self.collection: Collection = self.client.get_or_create_collection(
            name=collection_name,
            configuration={"hnsw": {"space": "cosine"}},
            embedding_function=None,
        )

    def __enter__(self) -> "ChromaKnowledgeRepository":
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.close()

    def close(self) -> None:
        self.client.close()

    def replace_document(self, chunks: list[KnowledgeChunk]) -> int:
        if not chunks:
            raise ValueError("至少需要一个知识库chunk")
        knowledge_base_id = validate_knowledge_base_id(
            chunks[0].knowledge_base_id
        )
        document_id = chunks[0].document_id
        if any(
            item.knowledge_base_id != knowledge_base_id
            or item.document_id != document_id
            for item in chunks
        ):
            raise ValueError("一次replace_document只能处理同一知识库的一份文档")

        existing = self.collection.get(
            where={
                "$and": [
                    {"knowledge_base_id": knowledge_base_id},
                    {"document_id": document_id},
                ]
            },
            include=[],
        )
        existing_ids = set(existing.get("ids", []))
        new_ids = [item.chunk_id for item in chunks]
        vectors = self.embeddings.embed_documents(
            [item.content for item in chunks]
        )
        self.collection.upsert(
            ids=new_ids,
            embeddings=vectors,
            documents=[item.content for item in chunks],
            metadatas=[item.metadata() for item in chunks],
        )
        stale_ids = sorted(existing_ids - set(new_ids))
        if stale_ids:
            self.collection.delete(ids=stale_ids)
        return len(existing_ids)

    def search(
        self,
        *,
        query: str,
        knowledge_base_id: str,
        top_k: int,
        relevance_threshold: float | None,
        document_ids: list[str] | None = None,
    ) -> KnowledgeSearchResponse:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("知识库查询不能为空")
        knowledge_base_id = validate_knowledge_base_id(knowledge_base_id)
        if top_k <= 0:
            raise ValueError("top_k必须是正整数")

        if document_ids is not None and not document_ids:
            return KnowledgeSearchResponse(
                status="no_evidence",
                knowledge_base_id=knowledge_base_id,
                results=[],
                message="未找到可靠证据",
            )
        query_vector = self.embeddings.embed_query(normalized_query)
        where: dict[str, Any] = {"knowledge_base_id": knowledge_base_id}
        if document_ids is not None:
            where = {
                "$and": [
                    {"knowledge_base_id": knowledge_base_id},
                    {"document_id": {"$in": document_ids}},
                ]
            }
        raw = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        results = self._to_results(raw, relevance_threshold)
        if not results:
            return KnowledgeSearchResponse(
                status="no_evidence",
                knowledge_base_id=knowledge_base_id,
                results=[],
                message="未找到可靠证据",
            )
        return KnowledgeSearchResponse(
            status="found",
            knowledge_base_id=knowledge_base_id,
            results=results,
            message=f"找到{len(results)}个相关片段",
        )

    @staticmethod
    def _to_results(
        raw: dict[str, Any],
        relevance_threshold: float | None,
    ) -> list[KnowledgeSearchResult]:
        ids = (raw.get("ids") or [[]])[0]
        documents = (raw.get("documents") or [[]])[0]
        metadatas = (raw.get("metadatas") or [[]])[0]
        distances = (raw.get("distances") or [[]])[0]
        results: list[KnowledgeSearchResult] = []
        for chunk_id, content, metadata, distance in zip(
            ids,
            documents,
            metadatas,
            distances,
            strict=False,
        ):
            if not isinstance(content, str) or not isinstance(metadata, dict):
                continue
            score = max(-1.0, min(1.0, 1.0 - float(distance)))
            if relevance_threshold is not None and score < relevance_threshold:
                continue
            raw_page = metadata.get("page", -1)
            page = int(raw_page) if int(raw_page) >= 0 else None
            results.append(
                KnowledgeSearchResult(
                    content=content,
                    score=round(score, 6),
                    source=str(metadata.get("source", "unknown")),
                    page=page,
                    document_id=str(metadata.get("document_id", "")),
                    chunk_id=str(metadata.get("chunk_id", chunk_id)),
                )
            )
        return results

    def count(
        self,
        *,
        knowledge_base_id: str,
        document_id: str | None = None,
    ) -> int:
        filters: list[dict[str, str]] = [
            {"knowledge_base_id": validate_knowledge_base_id(knowledge_base_id)}
        ]
        if document_id is not None:
            filters.append({"document_id": document_id})
        where: dict[str, Any] = (
            filters[0] if len(filters) == 1 else {"$and": filters}
        )
        return len(self.collection.get(where=where, include=[]).get("ids", []))

    def delete_document(
        self,
        *,
        knowledge_base_id: str,
        document_id: str,
    ) -> None:
        self.collection.delete(
            where={
                "$and": [
                    {"knowledge_base_id": validate_knowledge_base_id(knowledge_base_id)},
                    {"document_id": document_id},
                ]
            }
        )

    def delete_knowledge_base(self, *, knowledge_base_id: str) -> None:
        self.collection.delete(
            where={
                "knowledge_base_id": validate_knowledge_base_id(
                    knowledge_base_id
                )
            }
        )


class KnowledgeRetriever:
    def __init__(
        self,
        repository: ChromaKnowledgeRepository,
        *,
        default_top_k: int = 3,
        relevance_threshold: float | None = 0.35,
        ready_document_ids: Callable[[str], list[str]] | None = None,
    ):
        if default_top_k <= 0:
            raise ValueError("default_top_k必须是正整数")
        if relevance_threshold is not None and not 0.0 <= relevance_threshold <= 1.0:
            raise ValueError("relevance_threshold必须在0到1之间")
        self.repository = repository
        self.default_top_k = default_top_k
        self.relevance_threshold = relevance_threshold
        self.ready_document_ids = ready_document_ids

    def search(
        self,
        *,
        query: str,
        knowledge_base_id: str,
        top_k: int | None = None,
    ) -> KnowledgeSearchResponse:
        document_ids = None
        if self.ready_document_ids is not None:
            document_ids = list(self.ready_document_ids(knowledge_base_id))
        return self.repository.search(
            query=query,
            knowledge_base_id=knowledge_base_id,
            top_k=self.default_top_k if top_k is None else top_k,
            relevance_threshold=self.relevance_threshold,
            document_ids=document_ids,
        )
