from hashlib import sha256
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from ai_agent_learning.knowledge.loaders import loader_for
from ai_agent_learning.knowledge.models import (
    IngestionResult,
    KnowledgeChunk,
    validate_knowledge_base_id,
)
from ai_agent_learning.knowledge.repository import ChromaKnowledgeRepository


DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 120
CHINESE_FRIENDLY_SEPARATORS = [
    "\n\n",
    "\n",
    "。",
    "！",
    "？",
    "；",
    ". ",
    " ",
    "",
]


class KnowledgeIngestor:
    def __init__(
        self,
        repository: ChromaKnowledgeRepository,
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size必须是正整数")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap必须大于等于0且小于chunk_size")
        self.repository = repository
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=CHINESE_FRIENDLY_SEPARATORS,
            length_function=len,
        )

    def ingest_file(
        self,
        path: Path,
        *,
        knowledge_base_id: str,
        document_id: str | None = None,
        source_name: str | None = None,
    ) -> IngestionResult:
        path = path.expanduser().resolve()
        knowledge_base_id = validate_knowledge_base_id(knowledge_base_id)
        documents = loader_for(path).load()
        document_id = document_id or _document_id(knowledge_base_id, path)
        source_name = source_name or path.name
        chunks: list[KnowledgeChunk] = []
        chunk_sequence = 0
        for document in documents:
            page = document.metadata.get("page")
            for split in self.splitter.split_documents([document]):
                content = split.page_content.strip()
                if not content:
                    continue
                chunk_id = _chunk_id(
                    document_id,
                    page=page,
                    sequence=chunk_sequence,
                    content=content,
                )
                chunks.append(
                    KnowledgeChunk(
                        content=content,
                        knowledge_base_id=knowledge_base_id,
                        document_id=document_id,
                        source=source_name,
                        page=int(page) if page is not None else None,
                        chunk_id=chunk_id,
                    )
                )
                chunk_sequence += 1
        if not chunks:
            raise ValueError(f"文档切分后没有有效内容：{path.name}")
        replaced = self.repository.replace_document(chunks)
        return IngestionResult(
            knowledge_base_id=knowledge_base_id,
            document_id=document_id,
            source=source_name,
            chunk_count=len(chunks),
            replaced_chunk_count=replaced,
        )


def _document_id(knowledge_base_id: str, path: Path) -> str:
    source_key = path.resolve().as_posix().casefold()
    digest = sha256(f"{knowledge_base_id}\0{source_key}".encode()).hexdigest()
    return f"doc-{digest[:32]}"


def _chunk_id(
    document_id: str,
    *,
    page: int | None,
    sequence: int,
    content: str,
) -> str:
    digest = sha256(
        f"{document_id}\0{page}\0{sequence}\0{content}".encode()
    ).hexdigest()
    return f"chunk-{digest[:40]}"
