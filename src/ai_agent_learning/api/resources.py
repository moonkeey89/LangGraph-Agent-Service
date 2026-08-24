from collections.abc import Iterator
from contextlib import contextmanager

from ai_agent_learning.agents import build_supervisor_graph
from ai_agent_learning.auth import (
    AuthCookieConfig,
    AuthService,
    open_auth_catalog,
    resolve_auth_path,
)
from ai_agent_learning.api.service import AgentService
from ai_agent_learning.checkpoint import open_sqlite_checkpointer
from ai_agent_learning.config import Settings
from ai_agent_learning.embeddings import LocalModel2VecEmbeddings
from ai_agent_learning.llm import create_llm
from ai_agent_learning.logging_config import configure_logging
from ai_agent_learning.memory_store import open_sqlite_memory_store
from ai_agent_learning.knowledge import (
    ChromaKnowledgeRepository,
    KnowledgeIngestor,
    KnowledgeLibraryService,
    KnowledgeRetriever,
    open_knowledge_catalog,
    resolve_catalog_path,
    resolve_knowledge_directory,
    resolve_source_directory,
)
from ai_agent_learning.research import (
    ResearchExecutionService,
    ResearchService,
    build_research_graph,
    open_research_catalog,
    resolve_researchflow_path,
)


@contextmanager
def open_agent_service(settings: Settings | None = None) -> Iterator[AgentService]:
    """Own every process-wide Agent resource for one FastAPI lifespan."""
    active_settings = settings or Settings()
    configure_logging(active_settings.log_level)
    llm = create_llm(active_settings)
    embeddings = LocalModel2VecEmbeddings(
        active_settings.memory_embedding_model
    )
    with (
        open_sqlite_checkpointer() as checkpointer,
        open_sqlite_memory_store(
            embeddings=embeddings,
            dimensions=active_settings.memory_embedding_dimensions,
        ) as store,
        ChromaKnowledgeRepository(
            persist_directory=resolve_knowledge_directory(
                active_settings.knowledge_chroma_directory
            ),
            embeddings=embeddings,
        ) as knowledge_repository,
        open_knowledge_catalog(
            resolve_catalog_path(active_settings.knowledge_catalog_path)
        ) as knowledge_catalog,
        open_research_catalog(
            resolve_researchflow_path(
                active_settings.researchflow_database_path
            )
        ) as research_catalog,
        open_auth_catalog(
            resolve_auth_path(active_settings.auth_database_path)
        ) as auth_catalog,
    ):
        auth_service = AuthService(
            auth_catalog,
            AuthCookieConfig(
                session_cookie_name=active_settings.auth_session_cookie_name,
                csrf_cookie_name=active_settings.auth_csrf_cookie_name,
                csrf_header_name=active_settings.auth_csrf_header_name,
                secure=active_settings.auth_cookie_secure,
                same_site=active_settings.auth_cookie_samesite,
                domain=active_settings.auth_cookie_domain,
                session_ttl_minutes=active_settings.auth_session_ttl_minutes,
            ),
        )
        knowledge_ingestor = KnowledgeIngestor(
            knowledge_repository,
            chunk_size=active_settings.knowledge_chunk_size,
            chunk_overlap=active_settings.knowledge_chunk_overlap,
        )
        knowledge_service = KnowledgeLibraryService(
            catalog=knowledge_catalog,
            repository=knowledge_repository,
            ingestor=knowledge_ingestor,
            source_directory=resolve_source_directory(
                active_settings.knowledge_source_directory
            ),
            max_file_size_bytes=(
                active_settings.knowledge_upload_max_file_size_mb
                * 1024
                * 1024
            ),
            max_files_per_upload=(
                active_settings.knowledge_upload_max_files
            ),
        )
        knowledge_retriever = KnowledgeRetriever(
            knowledge_repository,
            default_top_k=active_settings.knowledge_top_k,
            relevance_threshold=(
                active_settings.knowledge_relevance_threshold
            ),
            ready_document_ids=knowledge_catalog.ready_document_ids,
        )
        research_service = ResearchService(
            research_catalog,
            knowledge_service,
        )
        graph = build_supervisor_graph(
            llm,
            checkpointer=checkpointer,
            store=store,
            memory_confidence_threshold=(
                active_settings.memory_manager_confidence_threshold
            ),
            max_subagent_calls=(
                active_settings.supervisor_max_subagent_calls
            ),
            knowledge_retriever=knowledge_retriever,
            knowledge_base_id=active_settings.knowledge_base_id,
            knowledge_top_k=active_settings.knowledge_top_k,
        )
        research_graph = build_research_graph(
            llm,
            retriever=knowledge_retriever,
            checkpointer=checkpointer,
            top_k=active_settings.knowledge_top_k,
        )
        research_execution_service = ResearchExecutionService(
            research_service,
            research_graph,
        )
        yield AgentService(
            graph,
            knowledge_service=knowledge_service,
            research_service=research_service,
            research_execution_service=research_execution_service,
            auth_service=auth_service,
        )
