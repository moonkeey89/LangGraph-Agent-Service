import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock

from ai_agent_learning.research.models import (
    AgentRun,
    ArtifactSource,
    ResearchArtifact,
    ResearchProject,
    ResearchTask,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESEARCHFLOW_DB_PATH = PROJECT_ROOT / "data" / "researchflow.sqlite"
CURRENT_SCHEMA_VERSION = 5


class ActiveAgentRunError(RuntimeError):
    def __init__(self, run_id: str):
        super().__init__(run_id)
        self.run_id = run_id


def resolve_researchflow_path(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ResearchCatalog:
    """SQLite persistence for ResearchFlow domain records only."""

    def __init__(self, path: Path):
        resolved_path = resolve_researchflow_path(path)
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        self.path = resolved_path
        self.connection = sqlite3.connect(
            resolved_path,
            check_same_thread=False,
        )
        self.connection.row_factory = sqlite3.Row
        self._lock = RLock()
        self._apply_migrations()

    def close(self) -> None:
        with self._lock:
            self.connection.close()

    def _apply_migrations(self) -> None:
        with self._lock, self.connection:
            self.connection.execute("PRAGMA foreign_keys = ON")
            self.connection.execute("PRAGMA journal_mode = WAL")
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL
                )
                """
            )
            applied_versions = {
                int(row["version"])
                for row in self.connection.execute(
                    "SELECT version FROM schema_migrations"
                ).fetchall()
            }
            if 1 not in applied_versions:
                self.connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS research_projects (
                        project_id TEXT PRIMARY KEY,
                        owner_user_id TEXT NOT NULL,
                        name TEXT NOT NULL,
                        description TEXT NOT NULL,
                        research_question TEXT NOT NULL,
                        status TEXT NOT NULL CHECK (
                            status IN ('draft', 'active', 'archived')
                        ),
                        default_knowledge_base_id TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_research_projects_owner_updated
                        ON research_projects(owner_user_id, updated_at DESC);
                    CREATE INDEX IF NOT EXISTS idx_research_projects_owner_status
                        ON research_projects(owner_user_id, status);
                    """
                )
                self.connection.execute(
                    """
                    INSERT OR IGNORE INTO schema_migrations (version, applied_at)
                    VALUES (?, ?)
                    """,
                    (1, _now()),
                )
            if 2 not in applied_versions:
                self.connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS research_tasks (
                        task_id TEXT PRIMARY KEY,
                        project_id TEXT NOT NULL,
                        title TEXT NOT NULL,
                        objective TEXT NOT NULL,
                        task_type TEXT NOT NULL CHECK (
                            task_type IN (
                                'literature_review', 'analysis',
                                'synthesis', 'general'
                            )
                        ),
                        status TEXT NOT NULL CHECK (
                            status IN (
                                'pending', 'running', 'blocked',
                                'completed', 'failed', 'cancelled'
                            )
                        ),
                        acceptance_criteria TEXT NOT NULL,
                        result_summary TEXT,
                        error_message TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        started_at TEXT,
                        completed_at TEXT,
                        FOREIGN KEY (project_id)
                            REFERENCES research_projects(project_id)
                            ON DELETE RESTRICT
                    );
                    CREATE INDEX IF NOT EXISTS idx_research_tasks_project_updated
                        ON research_tasks(project_id, updated_at DESC);
                    CREATE INDEX IF NOT EXISTS idx_research_tasks_project_status
                        ON research_tasks(project_id, status);
                    """
                )
                self.connection.execute(
                    """
                    INSERT OR IGNORE INTO schema_migrations (version, applied_at)
                    VALUES (?, ?)
                    """,
                    (2, _now()),
                )
            if 3 not in applied_versions:
                self.connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS research_artifacts (
                        artifact_id TEXT PRIMARY KEY,
                        project_id TEXT NOT NULL,
                        task_id TEXT,
                        title TEXT NOT NULL,
                        artifact_type TEXT NOT NULL CHECK (
                            artifact_type IN (
                                'note', 'literature_review',
                                'analysis', 'report'
                            )
                        ),
                        content TEXT NOT NULL,
                        status TEXT NOT NULL CHECK (
                            status IN ('draft', 'final')
                        ),
                        created_by TEXT NOT NULL CHECK (
                            created_by IN ('user', 'agent')
                        ),
                        sources TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        finalized_at TEXT,
                        FOREIGN KEY (project_id)
                            REFERENCES research_projects(project_id)
                            ON DELETE RESTRICT,
                        FOREIGN KEY (task_id)
                            REFERENCES research_tasks(task_id)
                            ON DELETE RESTRICT
                    );
                    CREATE INDEX IF NOT EXISTS idx_research_artifacts_project_updated
                        ON research_artifacts(project_id, updated_at DESC);
                    CREATE INDEX IF NOT EXISTS idx_research_artifacts_task
                        ON research_artifacts(task_id);
                    CREATE INDEX IF NOT EXISTS idx_research_artifacts_project_status
                        ON research_artifacts(project_id, status);
                    CREATE INDEX IF NOT EXISTS idx_research_artifacts_project_type
                        ON research_artifacts(project_id, artifact_type);
                    """
                )
                self.connection.execute(
                    """
                    INSERT OR IGNORE INTO schema_migrations (version, applied_at)
                    VALUES (?, ?)
                    """,
                    (3, _now()),
                )
            if 4 not in applied_versions:
                self.connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS agent_runs (
                        run_id TEXT PRIMARY KEY,
                        task_id TEXT NOT NULL,
                        thread_id TEXT NOT NULL UNIQUE,
                        attempt_number INTEGER NOT NULL CHECK (
                            attempt_number >= 1
                        ),
                        status TEXT NOT NULL CHECK (
                            status IN (
                                'pending', 'running', 'interrupted',
                                'completed', 'failed', 'cancelled'
                            )
                        ),
                        final_artifact_id TEXT,
                        error_message TEXT,
                        created_at TEXT NOT NULL,
                        started_at TEXT,
                        finished_at TEXT,
                        updated_at TEXT NOT NULL,
                        UNIQUE (task_id, attempt_number),
                        FOREIGN KEY (task_id)
                            REFERENCES research_tasks(task_id)
                            ON DELETE RESTRICT,
                        FOREIGN KEY (final_artifact_id)
                            REFERENCES research_artifacts(artifact_id)
                            ON DELETE RESTRICT
                    );
                    CREATE INDEX IF NOT EXISTS idx_agent_runs_task_updated
                        ON agent_runs(task_id, updated_at DESC);
                    CREATE INDEX IF NOT EXISTS idx_agent_runs_status_updated
                        ON agent_runs(status, updated_at DESC);
                    CREATE INDEX IF NOT EXISTS idx_agent_runs_final_artifact
                        ON agent_runs(final_artifact_id);
                    """
                )
                self.connection.execute(
                    """
                    INSERT OR IGNORE INTO schema_migrations (version, applied_at)
                    VALUES (?, ?)
                    """,
                    (4, _now()),
                )
            if 5 not in applied_versions:
                self.connection.execute(
                    """
                    ALTER TABLE agent_runs
                    RENAME COLUMN final_artifact_id TO output_artifact_id
                    """
                )
                self.connection.execute(
                    """
                    ALTER TABLE agent_runs ADD COLUMN outcome TEXT CHECK (
                        outcome IS NULL OR outcome IN (
                            'completed', 'blocked', 'failed', 'needs_review'
                        )
                    )
                    """
                )
                self.connection.execute(
                    """
                    ALTER TABLE research_artifacts ADD COLUMN origin_run_id TEXT
                        REFERENCES agent_runs(run_id) ON DELETE RESTRICT
                    """
                )
                self.connection.execute(
                    "DROP INDEX IF EXISTS idx_agent_runs_final_artifact"
                )
                self.connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_agent_runs_output_artifact
                        ON agent_runs(output_artifact_id)
                    """
                )
                self.connection.execute(
                    """
                    CREATE UNIQUE INDEX IF NOT EXISTS
                        idx_research_artifacts_origin_run
                    ON research_artifacts(origin_run_id)
                    WHERE origin_run_id IS NOT NULL
                    """
                )
                self.connection.execute(
                    """
                    INSERT OR IGNORE INTO schema_migrations (version, applied_at)
                    VALUES (?, ?)
                    """,
                    (5, _now()),
                )

    def create(self, project: ResearchProject) -> ResearchProject:
        with self._lock, self.connection:
            self.connection.execute(
                """
                INSERT INTO research_projects (
                    project_id, owner_user_id, name, description,
                    research_question, status, default_knowledge_base_id,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    project.project_id,
                    project.owner_user_id,
                    project.name,
                    project.description,
                    project.research_question,
                    project.status,
                    project.default_knowledge_base_id,
                    project.created_at,
                    project.updated_at,
                ),
            )
        return project

    def list_by_owner(self, owner_user_id: str) -> list[ResearchProject]:
        with self._lock:
            rows = self.connection.execute(
                """
                SELECT * FROM research_projects
                WHERE owner_user_id = ?
                ORDER BY updated_at DESC, project_id
                """,
                (owner_user_id,),
            ).fetchall()
        return [_project_from_row(row) for row in rows]

    def get_by_id(self, project_id: str) -> ResearchProject | None:
        with self._lock:
            row = self.connection.execute(
                "SELECT * FROM research_projects WHERE project_id = ?",
                (project_id,),
            ).fetchone()
        return _project_from_row(row) if row is not None else None

    def update(self, project: ResearchProject) -> ResearchProject:
        with self._lock, self.connection:
            cursor = self.connection.execute(
                """
                UPDATE research_projects SET
                    name = ?, description = ?, research_question = ?,
                    status = ?, default_knowledge_base_id = ?, updated_at = ?
                WHERE project_id = ? AND owner_user_id = ?
                """,
                (
                    project.name,
                    project.description,
                    project.research_question,
                    project.status,
                    project.default_knowledge_base_id,
                    project.updated_at,
                    project.project_id,
                    project.owner_user_id,
                ),
            )
        if cursor.rowcount != 1:
            raise KeyError(project.project_id)
        return project

    def delete(self, project_id: str, owner_user_id: str) -> bool:
        with self._lock, self.connection:
            cursor = self.connection.execute(
                """
                DELETE FROM research_projects
                WHERE project_id = ? AND owner_user_id = ?
                """,
                (project_id, owner_user_id),
            )
        return cursor.rowcount == 1

    def has_tasks(self, project_id: str) -> bool:
        with self._lock:
            row = self.connection.execute(
                """
                SELECT 1 FROM research_tasks
                WHERE project_id = ? LIMIT 1
                """,
                (project_id,),
            ).fetchone()
        return row is not None

    def has_artifacts(self, project_id: str) -> bool:
        with self._lock:
            row = self.connection.execute(
                """
                SELECT 1 FROM research_artifacts
                WHERE project_id = ? LIMIT 1
                """,
                (project_id,),
            ).fetchone()
        return row is not None

    def has_task_artifacts(self, project_id: str, task_id: str) -> bool:
        with self._lock:
            row = self.connection.execute(
                """
                SELECT 1 FROM research_artifacts
                WHERE project_id = ? AND task_id = ? LIMIT 1
                """,
                (project_id, task_id),
            ).fetchone()
        return row is not None

    def has_runs(self, project_id: str) -> bool:
        with self._lock:
            row = self.connection.execute(
                """
                SELECT 1 FROM agent_runs AS runs
                JOIN research_tasks AS tasks ON tasks.task_id = runs.task_id
                WHERE tasks.project_id = ? LIMIT 1
                """,
                (project_id,),
            ).fetchone()
        return row is not None

    def has_task_runs(self, project_id: str, task_id: str) -> bool:
        with self._lock:
            row = self.connection.execute(
                """
                SELECT 1 FROM agent_runs AS runs
                JOIN research_tasks AS tasks ON tasks.task_id = runs.task_id
                WHERE tasks.project_id = ? AND runs.task_id = ? LIMIT 1
                """,
                (project_id, task_id),
            ).fetchone()
        return row is not None

    def has_artifact_runs(self, project_id: str, artifact_id: str) -> bool:
        with self._lock:
            row = self.connection.execute(
                """
                SELECT 1 FROM agent_runs AS runs
                JOIN research_tasks AS tasks ON tasks.task_id = runs.task_id
                WHERE tasks.project_id = ? AND runs.output_artifact_id = ?
                LIMIT 1
                """,
                (project_id, artifact_id),
            ).fetchone()
        return row is not None

    def create_task(self, task: ResearchTask) -> ResearchTask:
        with self._lock, self.connection:
            self.connection.execute(
                """
                INSERT INTO research_tasks (
                    task_id, project_id, title, objective, task_type, status,
                    acceptance_criteria, result_summary, error_message,
                    created_at, updated_at, started_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task.task_id,
                    task.project_id,
                    task.title,
                    task.objective,
                    task.task_type,
                    task.status,
                    _serialize_criteria(task.acceptance_criteria),
                    task.result_summary,
                    task.error_message,
                    task.created_at,
                    task.updated_at,
                    task.started_at,
                    task.completed_at,
                ),
            )
        return task

    def list_tasks(self, project_id: str) -> list[ResearchTask]:
        with self._lock:
            rows = self.connection.execute(
                """
                SELECT * FROM research_tasks
                WHERE project_id = ?
                ORDER BY updated_at DESC, task_id
                """,
                (project_id,),
            ).fetchall()
        return [_task_from_row(row) for row in rows]

    def get_task(self, project_id: str, task_id: str) -> ResearchTask | None:
        with self._lock:
            row = self.connection.execute(
                """
                SELECT * FROM research_tasks
                WHERE project_id = ? AND task_id = ?
                """,
                (project_id, task_id),
            ).fetchone()
        return _task_from_row(row) if row is not None else None

    def update_task(self, task: ResearchTask) -> ResearchTask:
        with self._lock, self.connection:
            cursor = self.connection.execute(
                """
                UPDATE research_tasks SET
                    title = ?, objective = ?, task_type = ?, status = ?,
                    acceptance_criteria = ?, result_summary = ?,
                    error_message = ?, updated_at = ?, started_at = ?,
                    completed_at = ?
                WHERE task_id = ? AND project_id = ?
                """,
                (
                    task.title,
                    task.objective,
                    task.task_type,
                    task.status,
                    _serialize_criteria(task.acceptance_criteria),
                    task.result_summary,
                    task.error_message,
                    task.updated_at,
                    task.started_at,
                    task.completed_at,
                    task.task_id,
                    task.project_id,
                ),
            )
        if cursor.rowcount != 1:
            raise KeyError(task.task_id)
        return task

    def delete_task(self, project_id: str, task_id: str) -> bool:
        with self._lock, self.connection:
            cursor = self.connection.execute(
                """
                DELETE FROM research_tasks
                WHERE project_id = ? AND task_id = ?
                """,
                (project_id, task_id),
            )
        return cursor.rowcount == 1

    def create_artifact(self, artifact: ResearchArtifact) -> ResearchArtifact:
        with self._lock, self.connection:
            self.connection.execute(
                """
                INSERT INTO research_artifacts (
                    artifact_id, project_id, task_id, title, artifact_type,
                    content, status, created_by, sources, origin_run_id,
                    created_at, updated_at, finalized_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    artifact.artifact_id,
                    artifact.project_id,
                    artifact.task_id,
                    artifact.title,
                    artifact.artifact_type,
                    artifact.content,
                    artifact.status,
                    artifact.created_by,
                    _serialize_sources(artifact.sources),
                    artifact.origin_run_id,
                    artifact.created_at,
                    artifact.updated_at,
                    artifact.finalized_at,
                ),
            )
        return artifact

    def list_artifacts(
        self,
        project_id: str,
        *,
        task_id: str | None = None,
        artifact_type: str | None = None,
        status: str | None = None,
    ) -> list[ResearchArtifact]:
        clauses = ["project_id = ?"]
        parameters: list[str] = [project_id]
        if task_id is not None:
            clauses.append("task_id = ?")
            parameters.append(task_id)
        if artifact_type is not None:
            clauses.append("artifact_type = ?")
            parameters.append(artifact_type)
        if status is not None:
            clauses.append("status = ?")
            parameters.append(status)
        query = (
            "SELECT * FROM research_artifacts WHERE "
            + " AND ".join(clauses)
            + " ORDER BY updated_at DESC, artifact_id"
        )
        with self._lock:
            rows = self.connection.execute(query, parameters).fetchall()
        return [_artifact_from_row(row) for row in rows]

    def get_artifact(
        self,
        project_id: str,
        artifact_id: str,
    ) -> ResearchArtifact | None:
        with self._lock:
            row = self.connection.execute(
                """
                SELECT * FROM research_artifacts
                WHERE project_id = ? AND artifact_id = ?
                """,
                (project_id, artifact_id),
            ).fetchone()
        return _artifact_from_row(row) if row is not None else None

    def get_artifact_by_origin_run_id(
        self,
        origin_run_id: str,
    ) -> ResearchArtifact | None:
        with self._lock:
            row = self.connection.execute(
                """
                SELECT * FROM research_artifacts WHERE origin_run_id = ?
                """,
                (origin_run_id,),
            ).fetchone()
        return _artifact_from_row(row) if row is not None else None

    def update_artifact(self, artifact: ResearchArtifact) -> ResearchArtifact:
        with self._lock, self.connection:
            cursor = self.connection.execute(
                """
                UPDATE research_artifacts SET
                    title = ?, artifact_type = ?, content = ?, status = ?,
                    sources = ?, updated_at = ?, finalized_at = ?
                WHERE artifact_id = ? AND project_id = ?
                """,
                (
                    artifact.title,
                    artifact.artifact_type,
                    artifact.content,
                    artifact.status,
                    _serialize_sources(artifact.sources),
                    artifact.updated_at,
                    artifact.finalized_at,
                    artifact.artifact_id,
                    artifact.project_id,
                ),
            )
        if cursor.rowcount != 1:
            raise KeyError(artifact.artifact_id)
        return artifact

    def delete_artifact(self, project_id: str, artifact_id: str) -> bool:
        with self._lock, self.connection:
            cursor = self.connection.execute(
                """
                DELETE FROM research_artifacts
                WHERE project_id = ? AND artifact_id = ?
                """,
                (project_id, artifact_id),
            )
        return cursor.rowcount == 1

    def create_run(
        self,
        *,
        run_id: str,
        task_id: str,
        thread_id: str,
        timestamp: str,
    ) -> AgentRun:
        """Atomically allocate the next per-task attempt and insert it."""
        with self._lock:
            try:
                self.connection.execute("BEGIN IMMEDIATE")
                row = self.connection.execute(
                    """
                    SELECT COALESCE(MAX(attempt_number), 0) + 1 AS next_attempt
                    FROM agent_runs WHERE task_id = ?
                    """,
                    (task_id,),
                ).fetchone()
                attempt_number = int(row["next_attempt"])
                self.connection.execute(
                    """
                    INSERT INTO agent_runs (
                        run_id, task_id, thread_id, attempt_number, status,
                        outcome, output_artifact_id, error_message, created_at,
                        started_at, finished_at, updated_at
                    ) VALUES (?, ?, ?, ?, 'pending', NULL, NULL, NULL, ?, NULL, NULL, ?)
                    """,
                    (
                        run_id,
                        task_id,
                        thread_id,
                        attempt_number,
                        timestamp,
                        timestamp,
                    ),
                )
                self.connection.commit()
            except Exception:
                self.connection.rollback()
                raise
        result = self.get_run(task_id, run_id)
        assert result is not None
        return result

    def get_active_run(self, task_id: str) -> AgentRun | None:
        with self._lock:
            row = self.connection.execute(
                """
                SELECT * FROM agent_runs
                WHERE task_id = ?
                  AND status IN ('pending', 'running', 'interrupted')
                ORDER BY attempt_number DESC LIMIT 1
                """,
                (task_id,),
            ).fetchone()
        return _run_from_row(row) if row is not None else None

    def start_task_run(
        self,
        *,
        project_id: str,
        task_id: str,
        run_id: str,
        thread_id: str,
        timestamp: str,
    ) -> tuple[ResearchTask, AgentRun]:
        """Transaction A: atomically move Task and its new Run to running."""
        with self._lock:
            try:
                self.connection.execute("BEGIN IMMEDIATE")
                task_row = self.connection.execute(
                    """
                    SELECT * FROM research_tasks
                    WHERE project_id = ? AND task_id = ?
                    """,
                    (project_id, task_id),
                ).fetchone()
                if task_row is None:
                    raise ValueError("task_not_pending")
                active_row = self.connection.execute(
                    """
                    SELECT run_id FROM agent_runs
                    WHERE task_id = ?
                      AND status IN ('pending', 'running', 'interrupted')
                    ORDER BY attempt_number DESC LIMIT 1
                    """,
                    (task_id,),
                ).fetchone()
                if active_row is not None:
                    raise ActiveAgentRunError(str(active_row["run_id"]))
                if task_row["status"] != "pending":
                    raise ValueError("task_not_pending")
                next_row = self.connection.execute(
                    """
                    SELECT COALESCE(MAX(attempt_number), 0) + 1 AS value
                    FROM agent_runs WHERE task_id = ?
                    """,
                    (task_id,),
                ).fetchone()
                self.connection.execute(
                    """
                    INSERT INTO agent_runs (
                        run_id, task_id, thread_id, attempt_number, status,
                        outcome, output_artifact_id, error_message, created_at,
                        started_at, finished_at, updated_at
                    ) VALUES (?, ?, ?, ?, 'running', NULL, NULL, NULL,
                              ?, ?, NULL, ?)
                    """,
                    (
                        run_id,
                        task_id,
                        thread_id,
                        int(next_row["value"]),
                        timestamp,
                        timestamp,
                        timestamp,
                    ),
                )
                self.connection.execute(
                    """
                    UPDATE research_tasks SET
                        status = 'running', started_at = COALESCE(started_at, ?),
                        completed_at = NULL, result_summary = NULL,
                        error_message = NULL, updated_at = ?
                    WHERE project_id = ? AND task_id = ? AND status = 'pending'
                    """,
                    (timestamp, timestamp, project_id, task_id),
                )
                self.connection.commit()
            except Exception:
                self.connection.rollback()
                raise
        task = self.get_task(project_id, task_id)
        run = self.get_run(task_id, run_id)
        assert task is not None and run is not None
        return task, run

    def finish_task_run(
        self,
        *,
        task: ResearchTask,
        run: AgentRun,
        artifact: ResearchArtifact | None,
    ) -> tuple[ResearchTask, AgentRun, ResearchArtifact | None]:
        """Transaction B: atomically persist Artifact, Run and Task outcome."""
        with self._lock:
            try:
                self.connection.execute("BEGIN IMMEDIATE")
                current_row = self.connection.execute(
                    "SELECT * FROM agent_runs WHERE run_id = ? AND task_id = ?",
                    (run.run_id, run.task_id),
                ).fetchone()
                if current_row is None:
                    raise KeyError(run.run_id)
                current = _run_from_row(current_row)
                if current.status in {"completed", "failed", "cancelled"}:
                    existing = (
                        self._artifact_by_origin_locked(run.run_id)
                        if current.output_artifact_id is not None
                        else None
                    )
                    self.connection.commit()
                    persisted_task = self.get_task(task.project_id, task.task_id)
                    assert persisted_task is not None
                    return persisted_task, current, existing
                if current.status != "running":
                    raise ValueError("run_not_running")

                persisted_artifact = self._artifact_by_origin_locked(run.run_id)
                output_artifact_id = run.output_artifact_id
                if artifact is not None:
                    if artifact.origin_run_id != run.run_id:
                        raise ValueError("artifact_origin_mismatch")
                    if persisted_artifact is None:
                        self._insert_artifact_locked(artifact)
                        persisted_artifact = artifact
                    output_artifact_id = persisted_artifact.artifact_id
                elif persisted_artifact is not None:
                    output_artifact_id = persisted_artifact.artifact_id

                self.connection.execute(
                    """
                    UPDATE agent_runs SET
                        status = ?, outcome = ?, output_artifact_id = ?,
                        error_message = ?, started_at = ?, finished_at = ?,
                        updated_at = ?
                    WHERE run_id = ? AND task_id = ?
                    """,
                    (
                        run.status,
                        run.outcome,
                        output_artifact_id,
                        run.error_message,
                        run.started_at,
                        run.finished_at,
                        run.updated_at,
                        run.run_id,
                        run.task_id,
                    ),
                )
                self.connection.execute(
                    """
                    UPDATE research_tasks SET
                        status = ?, result_summary = ?, error_message = ?,
                        updated_at = ?, started_at = ?, completed_at = ?
                    WHERE project_id = ? AND task_id = ?
                    """,
                    (
                        task.status,
                        task.result_summary,
                        task.error_message,
                        task.updated_at,
                        task.started_at,
                        task.completed_at,
                        task.project_id,
                        task.task_id,
                    ),
                )
                self.connection.commit()
            except Exception:
                self.connection.rollback()
                raise
        persisted_task = self.get_task(task.project_id, task.task_id)
        persisted_run = self.get_run(run.task_id, run.run_id)
        assert persisted_task is not None and persisted_run is not None
        return persisted_task, persisted_run, persisted_artifact

    def _artifact_by_origin_locked(
        self,
        origin_run_id: str,
    ) -> ResearchArtifact | None:
        row = self.connection.execute(
            "SELECT * FROM research_artifacts WHERE origin_run_id = ?",
            (origin_run_id,),
        ).fetchone()
        return _artifact_from_row(row) if row is not None else None

    def _insert_artifact_locked(self, artifact: ResearchArtifact) -> None:
        self.connection.execute(
            """
            INSERT INTO research_artifacts (
                artifact_id, project_id, task_id, title, artifact_type,
                content, status, created_by, sources, origin_run_id,
                created_at, updated_at, finalized_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                artifact.artifact_id,
                artifact.project_id,
                artifact.task_id,
                artifact.title,
                artifact.artifact_type,
                artifact.content,
                artifact.status,
                artifact.created_by,
                _serialize_sources(artifact.sources),
                artifact.origin_run_id,
                artifact.created_at,
                artifact.updated_at,
                artifact.finalized_at,
            ),
        )

    def list_runs(self, task_id: str) -> list[AgentRun]:
        with self._lock:
            rows = self.connection.execute(
                """
                SELECT * FROM agent_runs
                WHERE task_id = ?
                ORDER BY attempt_number DESC
                """,
                (task_id,),
            ).fetchall()
        return [_run_from_row(row) for row in rows]

    def get_run(self, task_id: str, run_id: str) -> AgentRun | None:
        with self._lock:
            row = self.connection.execute(
                """
                SELECT * FROM agent_runs
                WHERE task_id = ? AND run_id = ?
                """,
                (task_id, run_id),
            ).fetchone()
        return _run_from_row(row) if row is not None else None

    def update_run(self, run: AgentRun) -> AgentRun:
        with self._lock, self.connection:
            cursor = self.connection.execute(
                """
                UPDATE agent_runs SET
                    status = ?, outcome = ?, output_artifact_id = ?, error_message = ?,
                    started_at = ?, finished_at = ?, updated_at = ?
                WHERE run_id = ? AND task_id = ?
                """,
                (
                    run.status,
                    run.outcome,
                    run.output_artifact_id,
                    run.error_message,
                    run.started_at,
                    run.finished_at,
                    run.updated_at,
                    run.run_id,
                    run.task_id,
                ),
            )
        if cursor.rowcount != 1:
            raise KeyError(run.run_id)
        return run


@contextmanager
def open_research_catalog(path: Path) -> Iterator[ResearchCatalog]:
    catalog = ResearchCatalog(path)
    try:
        yield catalog
    finally:
        catalog.close()


def _project_from_row(row: sqlite3.Row) -> ResearchProject:
    return ResearchProject(**dict(row))


def _serialize_criteria(criteria: list[str]) -> str:
    return json.dumps(criteria, ensure_ascii=False, separators=(",", ":"))


def _task_from_row(row: sqlite3.Row) -> ResearchTask:
    data = dict(row)
    try:
        criteria = json.loads(str(data["acceptance_criteria"]))
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise sqlite3.DataError("invalid acceptance_criteria JSON") from error
    if not isinstance(criteria, list) or not all(
        isinstance(item, str) for item in criteria
    ):
        raise sqlite3.DataError("invalid acceptance_criteria value")
    data["acceptance_criteria"] = criteria
    return ResearchTask(**data)


def _serialize_sources(sources: list[ArtifactSource]) -> str:
    values = [
        {
            "knowledge_base_id": item.knowledge_base_id,
            "document_id": item.document_id,
            "chunk_id": item.chunk_id,
            "source": item.source,
            "page": item.page,
            "excerpt": item.excerpt,
        }
        for item in sources
    ]
    return json.dumps(values, ensure_ascii=False, separators=(",", ":"))


def _artifact_from_row(row: sqlite3.Row) -> ResearchArtifact:
    data = dict(row)
    try:
        raw_sources = json.loads(str(data["sources"]))
        if not isinstance(raw_sources, list):
            raise TypeError
        sources = [ArtifactSource(**item) for item in raw_sources]
    except (TypeError, ValueError, KeyError) as error:
        raise sqlite3.DataError("invalid artifact sources JSON") from error
    data["sources"] = sources
    return ResearchArtifact(**data)


def _run_from_row(row: sqlite3.Row) -> AgentRun:
    return AgentRun(**dict(row))
