import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from uuid import uuid4

from ai_agent_learning.knowledge.models import (
    KnowledgeBaseRecord,
    KnowledgeDocumentRecord,
    validate_knowledge_base_id,
)
from ai_agent_learning.knowledge.repository import PROJECT_ROOT


KNOWLEDGE_CATALOG_PATH = PROJECT_ROOT / "data" / "knowledge_catalog.sqlite"


def resolve_catalog_path(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class KnowledgeCatalog:
    """SQLite management catalog; Chroma remains responsible only for chunks."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self._lock = RLock()
        self._create_schema()

    def close(self) -> None:
        self.connection.close()

    def _create_schema(self) -> None:
        with self._lock, self.connection:
            self.connection.execute("PRAGMA foreign_keys = ON")
            self.connection.execute("PRAGMA journal_mode = WAL")
            self.connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS knowledge_bases (
                    knowledge_base_id TEXT PRIMARY KEY,
                    owner_user_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_knowledge_bases_owner
                    ON knowledge_bases(owner_user_id);

                CREATE TABLE IF NOT EXISTS knowledge_documents (
                    document_id TEXT PRIMARY KEY,
                    knowledge_base_id TEXT NOT NULL,
                    original_filename TEXT NOT NULL,
                    stored_filename TEXT,
                    content_hash TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    size INTEGER NOT NULL,
                    status TEXT NOT NULL CHECK (
                        status IN ('processing', 'ready', 'failed')
                    ),
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    error_message TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (knowledge_base_id)
                        REFERENCES knowledge_bases(knowledge_base_id)
                        ON DELETE CASCADE,
                    UNIQUE (knowledge_base_id, content_hash)
                );
                CREATE INDEX IF NOT EXISTS idx_knowledge_documents_base
                    ON knowledge_documents(knowledge_base_id);
                """
            )

    def create_knowledge_base(
        self,
        *,
        owner_user_id: str,
        name: str,
        description: str,
        knowledge_base_id: str | None = None,
    ) -> KnowledgeBaseRecord:
        base_id = validate_knowledge_base_id(
            knowledge_base_id or f"kb_{uuid4().hex}"
        )
        timestamp = _now()
        with self._lock, self.connection:
            self.connection.execute(
                """
                INSERT INTO knowledge_bases (
                    knowledge_base_id, owner_user_id, name, description,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    base_id,
                    owner_user_id,
                    name,
                    description,
                    timestamp,
                    timestamp,
                ),
            )
        return self.get_owned_knowledge_base(base_id, owner_user_id)

    def ensure_cli_knowledge_base(
        self,
        *,
        knowledge_base_id: str,
        owner_user_id: str,
    ) -> KnowledgeBaseRecord:
        existing = self.get_owned_knowledge_base_or_none(
            knowledge_base_id,
            owner_user_id,
        )
        if existing is not None:
            return existing
        with self._lock:
            any_owner = self.connection.execute(
                "SELECT 1 FROM knowledge_bases WHERE knowledge_base_id = ?",
                (knowledge_base_id,),
            ).fetchone()
        if any_owner is not None:
            raise ValueError("知识库ID已由其他用户使用")
        return self.create_knowledge_base(
            owner_user_id=owner_user_id,
            name=knowledge_base_id,
            description="由CLI创建的知识库",
            knowledge_base_id=knowledge_base_id,
        )

    def list_knowledge_bases(self, owner_user_id: str) -> list[KnowledgeBaseRecord]:
        with self._lock:
            rows = self.connection.execute(
                """
                SELECT * FROM knowledge_bases
                WHERE owner_user_id = ?
                ORDER BY created_at DESC
                """,
                (owner_user_id,),
            ).fetchall()
        return [_base_record(row) for row in rows]

    def get_owned_knowledge_base_or_none(
        self,
        knowledge_base_id: str,
        owner_user_id: str,
    ) -> KnowledgeBaseRecord | None:
        validate_knowledge_base_id(knowledge_base_id)
        with self._lock:
            row = self.connection.execute(
                """
                SELECT * FROM knowledge_bases
                WHERE knowledge_base_id = ? AND owner_user_id = ?
                """,
                (knowledge_base_id, owner_user_id),
            ).fetchone()
        return _base_record(row) if row is not None else None

    def get_owned_knowledge_base(
        self,
        knowledge_base_id: str,
        owner_user_id: str,
    ) -> KnowledgeBaseRecord:
        result = self.get_owned_knowledge_base_or_none(
            knowledge_base_id,
            owner_user_id,
        )
        if result is None:
            raise KeyError(knowledge_base_id)
        return result

    def delete_knowledge_base_record(self, knowledge_base_id: str) -> None:
        with self._lock, self.connection:
            self.connection.execute(
                "DELETE FROM knowledge_bases WHERE knowledge_base_id = ?",
                (knowledge_base_id,),
            )

    def touch_knowledge_base(self, knowledge_base_id: str) -> None:
        with self._lock, self.connection:
            self.connection.execute(
                """
                UPDATE knowledge_bases SET updated_at = ?
                WHERE knowledge_base_id = ?
                """,
                (_now(), knowledge_base_id),
            )

    def list_documents(self, knowledge_base_id: str) -> list[KnowledgeDocumentRecord]:
        with self._lock:
            rows = self.connection.execute(
                """
                SELECT * FROM knowledge_documents
                WHERE knowledge_base_id = ?
                ORDER BY created_at DESC
                """,
                (knowledge_base_id,),
            ).fetchall()
        return [_document_record(row) for row in rows]

    def get_document(self, document_id: str) -> KnowledgeDocumentRecord | None:
        with self._lock:
            row = self.connection.execute(
                "SELECT * FROM knowledge_documents WHERE document_id = ?",
                (document_id,),
            ).fetchone()
        return _document_record(row) if row is not None else None

    def find_document_by_hash(
        self,
        *,
        knowledge_base_id: str,
        content_hash: str,
    ) -> KnowledgeDocumentRecord | None:
        with self._lock:
            row = self.connection.execute(
                """
                SELECT * FROM knowledge_documents
                WHERE knowledge_base_id = ? AND content_hash = ?
                """,
                (knowledge_base_id, content_hash),
            ).fetchone()
        return _document_record(row) if row is not None else None

    def save_processing_document(
        self,
        *,
        document_id: str,
        knowledge_base_id: str,
        original_filename: str,
        stored_filename: str,
        content_hash: str,
        content_type: str,
        size: int,
    ) -> KnowledgeDocumentRecord:
        timestamp = _now()
        existing = self.get_document(document_id)
        created_at = existing.created_at if existing is not None else timestamp
        with self._lock, self.connection:
            self.connection.execute(
                """
                INSERT INTO knowledge_documents (
                    document_id, knowledge_base_id, original_filename,
                    stored_filename, content_hash, content_type, size,
                    status, chunk_count, error_message, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'processing', 0, NULL, ?, ?)
                ON CONFLICT(document_id) DO UPDATE SET
                    original_filename=excluded.original_filename,
                    stored_filename=excluded.stored_filename,
                    content_type=excluded.content_type,
                    size=excluded.size,
                    status='processing', chunk_count=0,
                    error_message=NULL, updated_at=excluded.updated_at
                """,
                (
                    document_id,
                    knowledge_base_id,
                    original_filename,
                    stored_filename,
                    content_hash,
                    content_type,
                    size,
                    created_at,
                    timestamp,
                ),
            )
        result = self.get_document(document_id)
        assert result is not None
        return result

    def mark_document_ready(
        self,
        document_id: str,
        chunk_count: int,
    ) -> KnowledgeDocumentRecord:
        with self._lock, self.connection:
            self.connection.execute(
                """
                UPDATE knowledge_documents
                SET status='ready', chunk_count=?, error_message=NULL,
                    updated_at=?
                WHERE document_id=?
                """,
                (chunk_count, _now(), document_id),
            )
        result = self.get_document(document_id)
        assert result is not None
        return result

    def mark_document_failed(
        self,
        document_id: str,
        error_message: str,
    ) -> KnowledgeDocumentRecord:
        with self._lock, self.connection:
            self.connection.execute(
                """
                UPDATE knowledge_documents
                SET status='failed', stored_filename=NULL, chunk_count=0,
                    error_message=?, updated_at=?
                WHERE document_id=?
                """,
                (error_message, _now(), document_id),
            )
        result = self.get_document(document_id)
        assert result is not None
        return result

    def delete_document_record(self, document_id: str) -> None:
        with self._lock, self.connection:
            self.connection.execute(
                "DELETE FROM knowledge_documents WHERE document_id = ?",
                (document_id,),
            )

    def ready_document_ids(self, knowledge_base_id: str) -> list[str]:
        with self._lock:
            rows = self.connection.execute(
                """
                SELECT document_id FROM knowledge_documents
                WHERE knowledge_base_id = ? AND status = 'ready'
                ORDER BY document_id
                """,
                (knowledge_base_id,),
            ).fetchall()
        return [str(row["document_id"]) for row in rows]


@contextmanager
def open_knowledge_catalog(path: Path) -> Iterator[KnowledgeCatalog]:
    catalog = KnowledgeCatalog(resolve_catalog_path(path))
    try:
        yield catalog
    finally:
        catalog.close()


def _base_record(row: sqlite3.Row) -> KnowledgeBaseRecord:
    return KnowledgeBaseRecord(**dict(row))


def _document_record(row: sqlite3.Row) -> KnowledgeDocumentRecord:
    return KnowledgeDocumentRecord(**dict(row))
