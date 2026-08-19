import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ai_agent_learning.checkpoint import open_sqlite_checkpointer


class CheckpointTests(unittest.TestCase):
    def test_open_sqlite_checkpointer_creates_directory_and_closes_connection(self):
        with TemporaryDirectory() as temporary_directory:
            database_path = (
                Path(temporary_directory) / "nested" / "checkpoints.sqlite"
            )

            with open_sqlite_checkpointer(database_path) as checkpointer:
                connection = checkpointer.conn
                self.assertTrue(database_path.is_file())

            with self.assertRaises(sqlite3.ProgrammingError):
                connection.cursor()


if __name__ == "__main__":
    unittest.main()
