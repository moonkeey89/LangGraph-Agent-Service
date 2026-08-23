from collections.abc import Iterator
from contextlib import contextmanager

from ai_agent_learning.agents import build_supervisor_graph
from ai_agent_learning.api.service import AgentService
from ai_agent_learning.checkpoint import open_sqlite_checkpointer
from ai_agent_learning.config import Settings
from ai_agent_learning.embeddings import LocalModel2VecEmbeddings
from ai_agent_learning.llm import create_llm
from ai_agent_learning.logging_config import configure_logging
from ai_agent_learning.memory_store import open_sqlite_memory_store
from ai_agent_learning.knowledge import (
    ChromaKnowledgeRepository,
    KnowledgeRetriever,
    resolve_knowledge_directory,
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
    ):
        knowledge_retriever = KnowledgeRetriever(
            knowledge_repository,
            default_top_k=active_settings.knowledge_top_k,
            relevance_threshold=(
                active_settings.knowledge_relevance_threshold
            ),
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
        yield AgentService(graph)
