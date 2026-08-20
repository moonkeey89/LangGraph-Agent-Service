from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from langchain_core.embeddings import Embeddings
from langgraph.store.sqlite import SqliteStore

from ai_agent_learning.embeddings import DEFAULT_EMBEDDING_DIMENSIONS


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MEMORY_DB_PATH = PROJECT_ROOT / "data" / "memories.sqlite"


@contextmanager
def open_sqlite_memory_store(
    database_path: Path = MEMORY_DB_PATH,
    *,
    embeddings: Embeddings,
    dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
) -> Iterator[SqliteStore]:
    """Open the persistent long-term memory store without replacing its data."""
    database_path.parent.mkdir(parents=True, exist_ok=True)

    with SqliteStore.from_conn_string(
        str(database_path),
        index={
            "dims": dimensions,
            "embed": embeddings,
            "fields": ["content"],
        },
    ) as store:
        store.setup()
        yield store

