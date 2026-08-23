import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import BinaryIO
from uuid import uuid4

from ai_agent_learning.knowledge.catalog import KnowledgeCatalog
from ai_agent_learning.knowledge.ingestion import KnowledgeIngestor
from ai_agent_learning.knowledge.loaders import KnowledgeDocumentError
from ai_agent_learning.knowledge.models import (
    KnowledgeBaseRecord,
    KnowledgeDocumentRecord,
)
from ai_agent_learning.knowledge.repository import (
    ChromaKnowledgeRepository,
    PROJECT_ROOT,
)


logger = logging.getLogger(__name__)
SUPPORTED_UPLOAD_TYPES = {
    ".txt": {"text/plain", "application/octet-stream"},
    ".md": {"text/markdown", "text/plain", "application/octet-stream"},
    ".markdown": {
        "text/markdown",
        "text/plain",
        "application/octet-stream",
    },
    ".pdf": {"application/pdf", "application/octet-stream"},
}


class KnowledgeServiceError(RuntimeError):
    status_code = 400
    public_message = "Knowledge base request could not be completed"


class KnowledgeNotFoundError(KnowledgeServiceError):
    status_code = 404
    public_message = "Knowledge base resource was not found"


class KnowledgeValidationError(KnowledgeServiceError):
    status_code = 422

    def __init__(self, message: str):
        super().__init__(message)
        self.public_message = message


@dataclass(frozen=True)
class UploadCandidate:
    filename: str
    content_type: str
    stream: BinaryIO


@dataclass(frozen=True)
class UploadResult:
    document: KnowledgeDocumentRecord
    duplicate: bool


def resolve_source_directory(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


class KnowledgeLibraryService:
    """Coordinates catalog, controlled source files and the shared ingestor."""

    def __init__(
        self,
        *,
        catalog: KnowledgeCatalog,
        repository: ChromaKnowledgeRepository,
        ingestor: KnowledgeIngestor,
        source_directory: Path,
        max_file_size_bytes: int,
        max_files_per_upload: int,
    ):
        self.catalog = catalog
        self.repository = repository
        self.ingestor = ingestor
        self.source_directory = resolve_source_directory(source_directory)
        self.source_directory.mkdir(parents=True, exist_ok=True)
        self.max_file_size_bytes = max_file_size_bytes
        self.max_files_per_upload = max_files_per_upload
        self._mutation_lock = RLock()

    def create_knowledge_base(
        self,
        *,
        owner_user_id: str,
        name: str,
        description: str = "",
    ) -> KnowledgeBaseRecord:
        normalized_name = name.strip()
        normalized_description = description.strip()
        if not normalized_name or len(normalized_name) > 100:
            raise KnowledgeValidationError("知识库名称长度必须在1到100之间")
        if len(normalized_description) > 1000:
            raise KnowledgeValidationError("知识库描述不能超过1000个字符")
        return self.catalog.create_knowledge_base(
            owner_user_id=owner_user_id,
            name=normalized_name,
            description=normalized_description,
        )

    def list_knowledge_bases(self, owner_user_id: str) -> list[KnowledgeBaseRecord]:
        return self.catalog.list_knowledge_bases(owner_user_id)

    def get_knowledge_base(
        self,
        knowledge_base_id: str,
        owner_user_id: str,
    ) -> KnowledgeBaseRecord:
        try:
            return self.catalog.get_owned_knowledge_base(
                knowledge_base_id,
                owner_user_id,
            )
        except (KeyError, ValueError) as error:
            raise KnowledgeNotFoundError from error

    def ensure_owned(self, knowledge_base_id: str, owner_user_id: str) -> None:
        self.get_knowledge_base(knowledge_base_id, owner_user_id)

    def list_documents(
        self,
        *,
        knowledge_base_id: str,
        owner_user_id: str,
    ) -> list[KnowledgeDocumentRecord]:
        self.ensure_owned(knowledge_base_id, owner_user_id)
        return self.catalog.list_documents(knowledge_base_id)

    def upload_documents(
        self,
        *,
        knowledge_base_id: str,
        owner_user_id: str,
        uploads: list[UploadCandidate],
    ) -> list[UploadResult]:
        self.ensure_owned(knowledge_base_id, owner_user_id)
        if not uploads:
            raise KnowledgeValidationError("至少选择一个文件")
        if len(uploads) > self.max_files_per_upload:
            raise KnowledgeValidationError(
                f"单次最多上传{self.max_files_per_upload}个文件"
            )
        with self._mutation_lock:
            return [
                self._upload_one(knowledge_base_id, candidate)
                for candidate in uploads
            ]

    def _upload_one(
        self,
        knowledge_base_id: str,
        upload: UploadCandidate,
    ) -> UploadResult:
        original_name, suffix = _safe_filename(upload.filename)
        _validate_declared_type(suffix, upload.content_type)
        temporary_directory = self.source_directory / ".tmp"
        temporary_directory.mkdir(parents=True, exist_ok=True)
        temporary_path = temporary_directory / f"{uuid4().hex}.upload"
        content_hash = hashlib.sha256()
        size = 0
        prefix = bytearray()
        try:
            with temporary_path.open("xb") as output:
                while True:
                    chunk = upload.stream.read(64 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > self.max_file_size_bytes:
                        raise KnowledgeValidationError(
                            f"文件超过{self.max_file_size_bytes // (1024 * 1024)}MB限制"
                        )
                    if len(prefix) < 4096:
                        prefix.extend(chunk[: 4096 - len(prefix)])
                    content_hash.update(chunk)
                    output.write(chunk)
            if size == 0:
                raise KnowledgeValidationError("不能上传空文件")
            _validate_signature(suffix, bytes(prefix))
            digest = content_hash.hexdigest()
            duplicate = self.catalog.find_document_by_hash(
                knowledge_base_id=knowledge_base_id,
                content_hash=digest,
            )
            if duplicate is not None and duplicate.status == "ready":
                temporary_path.unlink(missing_ok=True)
                return UploadResult(document=duplicate, duplicate=True)

            document_id = _document_id(knowledge_base_id, digest)
            relative_name = f"{knowledge_base_id}/{document_id}{suffix}"
            final_path = self._controlled_path(relative_name)
            final_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path.replace(final_path)
            self.catalog.save_processing_document(
                document_id=document_id,
                knowledge_base_id=knowledge_base_id,
                original_filename=original_name,
                stored_filename=relative_name,
                content_hash=digest,
                content_type=upload.content_type or "application/octet-stream",
                size=size,
            )
            try:
                ingestion = self.ingestor.ingest_file(
                    final_path,
                    knowledge_base_id=knowledge_base_id,
                    document_id=document_id,
                    source_name=original_name,
                )
                document = self.catalog.mark_document_ready(
                    document_id,
                    ingestion.chunk_count,
                )
                self.catalog.touch_knowledge_base(knowledge_base_id)
                return UploadResult(document=document, duplicate=False)
            except Exception as error:
                logger.exception("Knowledge document indexing failed")
                self.repository.delete_document(
                    knowledge_base_id=knowledge_base_id,
                    document_id=document_id,
                )
                final_path.unlink(missing_ok=True)
                message = _safe_ingestion_error(error)
                document = self.catalog.mark_document_failed(
                    document_id,
                    message,
                )
                self.catalog.touch_knowledge_base(knowledge_base_id)
                return UploadResult(document=document, duplicate=False)
        finally:
            temporary_path.unlink(missing_ok=True)

    def delete_document(
        self,
        *,
        knowledge_base_id: str,
        document_id: str,
        owner_user_id: str,
    ) -> None:
        self.ensure_owned(knowledge_base_id, owner_user_id)
        document = self.catalog.get_document(document_id)
        if document is None or document.knowledge_base_id != knowledge_base_id:
            raise KnowledgeNotFoundError
        with self._mutation_lock:
            self.repository.delete_document(
                knowledge_base_id=knowledge_base_id,
                document_id=document_id,
            )
            if document.stored_filename:
                self._controlled_path(document.stored_filename).unlink(
                    missing_ok=True
                )
            self.catalog.delete_document_record(document_id)
            self.catalog.touch_knowledge_base(knowledge_base_id)

    def delete_knowledge_base(
        self,
        *,
        knowledge_base_id: str,
        owner_user_id: str,
    ) -> None:
        self.ensure_owned(knowledge_base_id, owner_user_id)
        with self._mutation_lock:
            documents = self.catalog.list_documents(knowledge_base_id)
            self.repository.delete_knowledge_base(
                knowledge_base_id=knowledge_base_id
            )
            for document in documents:
                if document.stored_filename:
                    self._controlled_path(document.stored_filename).unlink(
                        missing_ok=True
                    )
            base_directory = self._controlled_path(knowledge_base_id)
            if base_directory.exists():
                try:
                    base_directory.rmdir()
                except OSError:
                    logger.warning(
                        "Knowledge source directory was not empty after cleanup"
                    )
            self.catalog.delete_knowledge_base_record(knowledge_base_id)

    def ingest_cli_file(
        self,
        *,
        path: Path,
        knowledge_base_id: str,
        owner_user_id: str,
    ) -> UploadResult:
        self.catalog.ensure_cli_knowledge_base(
            knowledge_base_id=knowledge_base_id,
            owner_user_id=owner_user_id,
        )
        content_type = _content_type_for_suffix(path.suffix.casefold())
        with self._mutation_lock, path.open("rb") as stream:
            return self._upload_one(
                knowledge_base_id,
                UploadCandidate(path.name, content_type, stream),
            )

    def _controlled_path(self, relative_name: str) -> Path:
        candidate = (self.source_directory / relative_name).resolve()
        if not candidate.is_relative_to(self.source_directory):
            raise KnowledgeValidationError("非法的文件存储路径")
        return candidate


def _safe_filename(filename: str) -> tuple[str, str]:
    normalized = filename.replace("\\", "/")
    basename = Path(normalized).name.strip()
    if not basename or basename in {".", ".."}:
        raise KnowledgeValidationError("文件名无效")
    suffix = Path(basename).suffix.casefold()
    if suffix not in SUPPORTED_UPLOAD_TYPES:
        raise KnowledgeValidationError("仅支持TXT、Markdown和PDF文件")
    display_name = "".join(
        character if character.isprintable() and character not in "\x00" else "_"
        for character in basename
    )[:255]
    return display_name, suffix


def _validate_declared_type(suffix: str, content_type: str) -> None:
    normalized = (content_type or "application/octet-stream").split(";", 1)[0]
    if normalized.casefold() not in SUPPORTED_UPLOAD_TYPES[suffix]:
        raise KnowledgeValidationError("文件扩展名与Content-Type不匹配")


def _validate_signature(suffix: str, prefix: bytes) -> None:
    if suffix == ".pdf":
        if not prefix.startswith(b"%PDF-"):
            raise KnowledgeValidationError("PDF文件签名无效")
        return
    if b"\x00" in prefix:
        raise KnowledgeValidationError("文本文件包含二进制内容")
    try:
        prefix.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise KnowledgeValidationError("文本文件必须使用UTF-8编码") from error


def _document_id(knowledge_base_id: str, content_hash: str) -> str:
    digest = hashlib.sha256(
        f"{knowledge_base_id}\0{content_hash}".encode()
    ).hexdigest()
    return f"doc-{digest[:32]}"


def _content_type_for_suffix(suffix: str) -> str:
    if suffix == ".pdf":
        return "application/pdf"
    if suffix in {".md", ".markdown"}:
        return "text/markdown"
    return "text/plain"


def _safe_ingestion_error(error: Exception) -> str:
    if isinstance(error, (KnowledgeDocumentError, KnowledgeValidationError)):
        message = str(error).strip()
        return message[:500] if message else "文档解析或索引失败"
    return "文档解析或索引失败"
