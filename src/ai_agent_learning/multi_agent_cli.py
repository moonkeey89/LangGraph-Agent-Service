import logging

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore
from pydantic import ValidationError

from ai_agent_learning.agents import build_supervisor_graph
from ai_agent_learning.checkpoint import CHECKPOINT_DB_PATH, open_sqlite_checkpointer
from ai_agent_learning.cli import prompt_thread_id, prompt_user_id, run_cli
from ai_agent_learning.config import Settings
from ai_agent_learning.embeddings import LocalModel2VecEmbeddings
from ai_agent_learning.llm import create_llm
from ai_agent_learning.logging_config import configure_logging
from ai_agent_learning.memory_store import MEMORY_DB_PATH, open_sqlite_memory_store


logger = logging.getLogger(__name__)


def create_supervisor_app(
    settings: Settings,
    checkpointer: BaseCheckpointSaver,
    store: BaseStore | None = None,
):
    llm = create_llm(settings)
    return build_supervisor_graph(
        llm,
        checkpointer=checkpointer,
        store=store,
        memory_confidence_threshold=(
            settings.memory_manager_confidence_threshold
        ),
        max_subagent_calls=settings.supervisor_max_subagent_calls,
    )


def main() -> int:
    try:
        settings = Settings()
    except ValidationError:
        configure_logging("ERROR")
        logger.error("配置无效，请检查 .env 中的 DeepSeek 配置")
        return 1

    configure_logging(settings.log_level)

    try:
        embeddings = LocalModel2VecEmbeddings(
            settings.memory_embedding_model
        )
        with (
            open_sqlite_checkpointer() as checkpointer,
            open_sqlite_memory_store(
                embeddings=embeddings,
                dimensions=settings.memory_embedding_dimensions,
            ) as store,
        ):
            app = create_supervisor_app(settings, checkpointer, store)

            user_id = prompt_user_id()
            if user_id is None:
                return 0
            thread_id = prompt_thread_id()
            if thread_id is None:
                return 0

            logger.info(
                "Supervisor Agent started with model %s, user %s, thread %s, "
                "checkpoint database %s, memory database %s, max handoffs %s",
                settings.deepseek_model,
                user_id,
                thread_id,
                CHECKPOINT_DB_PATH,
                MEMORY_DB_PATH,
                settings.supervisor_max_subagent_calls,
            )
            run_cli(app, thread_id, user_id)
    except Exception:
        logger.exception("Supervisor Agent startup failed")
        return 1

    return 0
