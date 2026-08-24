from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
import sqlite3
from threading import RLock

from ai_agent_learning.auth.models import Session, User


PROJECT_ROOT = Path(__file__).resolve().parents[3]
AUTH_DB_PATH = PROJECT_ROOT / "data" / "auth.sqlite"
SCHEMA_VERSION = 1


def resolve_auth_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


class AuthCatalog:
    def __init__(self, database_path: Path):
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self.database_path = database_path
        self.connection = sqlite3.connect(database_path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self._lock = RLock()
        self._initialize()

    def _initialize(self) -> None:
        with self._lock, self.connection:
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS auth_schema_migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL
                )
                """
            )
            row = self.connection.execute(
                "SELECT COALESCE(MAX(version), 0) AS version FROM auth_schema_migrations"
            ).fetchone()
            version = int(row["version"])
            if version < 1:
                self.connection.executescript(
                    """
                    CREATE TABLE users (
                        user_id TEXT PRIMARY KEY,
                        username TEXT NOT NULL UNIQUE,
                        email TEXT NOT NULL UNIQUE,
                        password_hash TEXT NOT NULL,
                        is_active INTEGER NOT NULL CHECK (is_active IN (0, 1)),
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    CREATE TABLE sessions (
                        session_id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        token_hash TEXT NOT NULL UNIQUE,
                        csrf_token_hash TEXT NOT NULL,
                        expires_at TEXT NOT NULL,
                        revoked_at TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE RESTRICT
                    );
                    CREATE INDEX idx_users_active ON users(is_active);
                    CREATE INDEX idx_sessions_user ON sessions(user_id);
                    CREATE INDEX idx_sessions_expires ON sessions(expires_at);
                    CREATE INDEX idx_sessions_revoked ON sessions(revoked_at);
                    """
                )
                self.connection.execute(
                    "INSERT INTO auth_schema_migrations(version, applied_at) VALUES (?, datetime('now'))",
                    (SCHEMA_VERSION,),
                )

    def create_user(self, user: User) -> User:
        with self._lock, self.connection:
            self.connection.execute(
                """
                INSERT INTO users(
                    user_id, username, email, password_hash, is_active,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user.user_id,
                    user.username,
                    user.email,
                    user.password_hash,
                    int(user.is_active),
                    user.created_at,
                    user.updated_at,
                ),
            )
        return user

    def get_user_by_id(self, user_id: str) -> User | None:
        with self._lock:
            row = self.connection.execute(
                "SELECT * FROM users WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        return _user(row) if row is not None else None

    def get_user_by_username(self, username: str) -> User | None:
        with self._lock:
            row = self.connection.execute(
                "SELECT * FROM users WHERE username = ?",
                (username,),
            ).fetchone()
        return _user(row) if row is not None else None

    def get_user_by_email(self, email: str) -> User | None:
        with self._lock:
            row = self.connection.execute(
                "SELECT * FROM users WHERE email = ?",
                (email,),
            ).fetchone()
        return _user(row) if row is not None else None

    def create_session(self, session: Session) -> Session:
        with self._lock, self.connection:
            self.connection.execute(
                """
                INSERT INTO sessions(
                    session_id, user_id, token_hash, csrf_token_hash,
                    expires_at, revoked_at, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session.session_id,
                    session.user_id,
                    session.token_hash,
                    session.csrf_token_hash,
                    session.expires_at,
                    session.revoked_at,
                    session.created_at,
                    session.updated_at,
                ),
            )
        return session

    def get_session_by_token_hash(self, token_hash: str) -> Session | None:
        with self._lock:
            row = self.connection.execute(
                "SELECT * FROM sessions WHERE token_hash = ?",
                (token_hash,),
            ).fetchone()
        return _session(row) if row is not None else None

    def revoke_session(self, session_id: str, revoked_at: str) -> None:
        with self._lock, self.connection:
            self.connection.execute(
                """
                UPDATE sessions
                SET revoked_at = COALESCE(revoked_at, ?), updated_at = ?
                WHERE session_id = ?
                """,
                (revoked_at, revoked_at, session_id),
            )

    def close(self) -> None:
        with self._lock:
            self.connection.close()


def _user(row: sqlite3.Row) -> User:
    return User(
        user_id=str(row["user_id"]),
        username=str(row["username"]),
        email=str(row["email"]),
        password_hash=str(row["password_hash"]),
        is_active=bool(row["is_active"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _session(row: sqlite3.Row) -> Session:
    return Session(
        session_id=str(row["session_id"]),
        user_id=str(row["user_id"]),
        token_hash=str(row["token_hash"]),
        csrf_token_hash=str(row["csrf_token_hash"]),
        expires_at=str(row["expires_at"]),
        revoked_at=str(row["revoked_at"]) if row["revoked_at"] else None,
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


@contextmanager
def open_auth_catalog(database_path: Path) -> Iterator[AuthCatalog]:
    catalog = AuthCatalog(resolve_auth_path(database_path))
    try:
        yield catalog
    finally:
        catalog.close()
