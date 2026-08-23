import argparse
import logging
from pathlib import Path

from pydantic import ValidationError

from ai_agent_learning.config import Settings
from ai_agent_learning.embeddings import LocalModel2VecEmbeddings
from ai_agent_learning.knowledge.ingestion import KnowledgeIngestor
from ai_agent_learning.knowledge.catalog import (
    open_knowledge_catalog,
    resolve_catalog_path,
)
from ai_agent_learning.knowledge.loaders import (
    discover_documents,
    KnowledgeDocumentError,
)
from ai_agent_learning.knowledge.models import validate_knowledge_base_id
from ai_agent_learning.knowledge.repository import (
    ChromaKnowledgeRepository,
    resolve_knowledge_directory,
)
from ai_agent_learning.knowledge.service import (
    KnowledgeLibraryService,
    resolve_source_directory,
)
from ai_agent_learning.logging_config import configure_logging


logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="将本地TXT、Markdown和PDF文档离线写入RAG知识库",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="一个或多个文件/目录；目录会递归发现受支持文档",
    )
    parser.add_argument(
        "--owner-user-id",
        default="default_user",
        help="知识库所有者；CLI默认使用default_user",
    )
    parser.add_argument(
        "--knowledge-base-id",
        dest="knowledge_base_id",
        help="目标知识库ID；默认使用Settings中的KNOWLEDGE_BASE_ID",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        settings = Settings()
        configure_logging(settings.log_level)
        knowledge_base_id = validate_knowledge_base_id(
            args.knowledge_base_id or settings.knowledge_base_id
        )
        embeddings = LocalModel2VecEmbeddings(
            settings.memory_embedding_model
        )
        with (
            ChromaKnowledgeRepository(
                persist_directory=resolve_knowledge_directory(
                    settings.knowledge_chroma_directory
                ),
                embeddings=embeddings,
            ) as repository,
            open_knowledge_catalog(
                resolve_catalog_path(settings.knowledge_catalog_path)
            ) as catalog,
        ):
            ingestor = KnowledgeIngestor(
                repository,
                chunk_size=settings.knowledge_chunk_size,
                chunk_overlap=settings.knowledge_chunk_overlap,
            )
            service = KnowledgeLibraryService(
                catalog=catalog,
                repository=repository,
                ingestor=ingestor,
                source_directory=resolve_source_directory(
                    settings.knowledge_source_directory
                ),
                max_file_size_bytes=(
                    settings.knowledge_upload_max_file_size_mb * 1024 * 1024
                ),
                max_files_per_upload=settings.knowledge_upload_max_files,
            )
            documents = discover_documents(args.paths)
            for path in documents:
                upload = service.ingest_cli_file(
                    path=path,
                    knowledge_base_id=knowledge_base_id,
                    owner_user_id=args.owner_user_id,
                )
                result = upload.document
                print(
                    f"{'已跳过重复文档' if upload.duplicate else '已索引'} "
                    f"{result.original_filename}: chunks={result.chunk_count}, "
                    f"status={result.status}, document_id={result.document_id}"
                )
        print(
            f"知识库 {knowledge_base_id} 入库完成，共处理{len(documents)}份文档。"
        )
        return 0
    except (KnowledgeDocumentError, ValueError, ValidationError) as error:
        logger.error("知识库入库失败：%s", error)
        return 2
    except Exception:
        logger.exception("知识库入库发生内部错误")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
