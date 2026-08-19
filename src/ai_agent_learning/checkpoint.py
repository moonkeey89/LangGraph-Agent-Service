from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DB_PATH = PROJECT_ROOT / "data" / "checkpoints.sqlite"


@contextmanager
def open_sqlite_checkpointer(
    database_path: Path = CHECKPOINT_DB_PATH,
) -> Iterator[SqliteSaver]:
    """打开持久化 Checkpointer，并在退出上下文时关闭连接。"""
    database_path.parent.mkdir(parents=True, exist_ok=True)

    with SqliteSaver.from_conn_string(str(database_path)) as checkpointer:
        checkpointer.setup()
        yield checkpointer
