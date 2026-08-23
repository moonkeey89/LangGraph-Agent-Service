import sqlite3
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

from ai_agent_learning.knowledge.models import KnowledgeChunk
from ai_agent_learning.knowledge.service import KnowledgeNotFoundError
from ai_agent_learning.research import (
    CURRENT_SCHEMA_VERSION,
    AgentRunConflictError,
    AgentRunNotFoundError,
    AgentRunValidationError,
    ResearchArtifactConflictError,
    ResearchArtifactNotFoundError,
    ResearchCatalog,
    ResearchProjectConflictError,
    ResearchProjectNotFoundError,
    ResearchService,
    ResearchTaskConflictError,
    ResearchTaskNotFoundError,
)


class KnowledgeStub:
    def ensure_owned(self, _knowledge_base_id: str, _owner_user_id: str) -> None:
        raise KnowledgeNotFoundError

    def get_ready_chunk(self, **_kwargs) -> KnowledgeChunk:
        raise KnowledgeNotFoundError


def _prepare_version_three_database(path: Path) -> tuple[str, str, str]:
    catalog = ResearchCatalog(path)
    service = ResearchService(catalog, KnowledgeStub())
    project = service.create_project(
        owner_user_id="user_001",
        name="已有项目",
        status="active",
    )
    task = service.create_task(
        project.project_id,
        "user_001",
        title="已有任务",
    )
    artifact = service.create_artifact(
        project.project_id,
        "user_001",
        task_id=task.task_id,
        title="已有成果",
        content="已有成果正文。",
    )
    catalog.close()
    connection = sqlite3.connect(path)
    try:
        connection.execute("DROP TABLE agent_runs")
        connection.execute("DELETE FROM schema_migrations WHERE version = 4")
        connection.commit()
    finally:
        connection.close()
    return project.project_id, task.task_id, artifact.artifact_id


class AgentRunMigrationTests(unittest.TestCase):
    def test_version_three_upgrades_to_four_and_preserves_domain_records(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "research.sqlite"
            project_id, task_id, artifact_id = _prepare_version_three_database(path)
            catalog = ResearchCatalog(path)
            try:
                versions = [
                    row[0]
                    for row in catalog.connection.execute(
                        "SELECT version FROM schema_migrations ORDER BY version"
                    )
                ]
                self.assertEqual(versions, [1, 2, 3, CURRENT_SCHEMA_VERSION])
                self.assertIsNotNone(catalog.get_by_id(project_id))
                self.assertIsNotNone(catalog.get_task(project_id, task_id))
                self.assertIsNotNone(catalog.get_artifact(project_id, artifact_id))
                self.assertEqual(
                    catalog.connection.execute("PRAGMA foreign_keys").fetchone()[0],
                    1,
                )
            finally:
                catalog.close()

            restarted = ResearchCatalog(path)
            try:
                self.assertEqual(
                    restarted.connection.execute(
                        "SELECT COUNT(*) FROM schema_migrations"
                    ).fetchone()[0],
                    4,
                )
            finally:
                restarted.close()


class AgentRunServiceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.path = Path(self.temporary.name) / "research.sqlite"
        self.catalog = ResearchCatalog(self.path)
        self.service = ResearchService(self.catalog, KnowledgeStub())
        self.project = self.service.create_project(
            owner_user_id="user_001",
            name="运行测试项目",
            status="active",
        )
        self.other_project = self.service.create_project(
            owner_user_id="user_001",
            name="其他项目",
            status="active",
        )
        self.foreign_project = self.service.create_project(
            owner_user_id="user_002",
            name="其他用户项目",
            status="active",
        )
        self.task = self.service.create_task(
            self.project.project_id,
            "user_001",
            title="主任务",
        )
        self.second_task = self.service.create_task(
            self.project.project_id,
            "user_001",
            title="第二任务",
        )
        self.other_task = self.service.create_task(
            self.other_project.project_id,
            "user_001",
            title="其他项目任务",
        )

    def tearDown(self):
        self.catalog.close()
        self.temporary.cleanup()

    def create_run(self, task_id: str | None = None):
        return self.service.create_run(
            self.project.project_id,
            task_id or self.task.task_id,
            "user_001",
        )

    def transition(self, run, target_status, **kwargs):
        return self.service.transition_run(
            self.project.project_id,
            run.task_id,
            run.run_id,
            "user_001",
            target_status=target_status,
            **kwargs,
        )

    def finalized_artifact(self, *, task_id=None, project=None):
        project = project or self.project
        artifact = self.service.create_artifact(
            project.project_id,
            "user_001",
            task_id=task_id,
            title="最终成果",
            content="最终成果正文。",
        )
        return self.service.finalize_artifact(
            project.project_id,
            artifact.artifact_id,
            "user_001",
        )

    def test_create_allocates_ids_attempts_and_per_task_sequences(self):
        first = self.create_run()
        second = self.create_run()
        other_task_first = self.create_run(self.second_task.task_id)
        self.assertTrue(first.run_id.startswith("run_"))
        self.assertTrue(first.thread_id.startswith("research-run-"))
        self.assertNotEqual(first.thread_id, second.thread_id)
        self.assertEqual((first.attempt_number, second.attempt_number), (1, 2))
        self.assertEqual(other_task_first.attempt_number, 1)
        self.assertEqual(first.status, "pending")
        self.assertIsNone(first.started_at)
        self.assertIsNone(first.finished_at)
        with self.assertRaises(sqlite3.IntegrityError):
            self.catalog.create_run(
                run_id="run_duplicate_thread",
                task_id=self.second_task.task_id,
                thread_id=first.thread_id,
                timestamp="2026-01-01T00:00:00+00:00",
            )
        with self.assertRaises(sqlite3.IntegrityError):
            self.catalog.create_run(
                run_id="run_missing_task",
                task_id="rt_missing",
                thread_id="research-run-missing-task",
                timestamp="2026-01-01T00:00:00+00:00",
            )

    def test_concurrent_attempt_allocation_is_unique(self):
        catalogs = [ResearchCatalog(self.path) for _ in range(5)]

        def create(index: int):
            service = ResearchService(catalogs[index], KnowledgeStub())
            return service.create_run(
                self.project.project_id,
                self.task.task_id,
                "user_001",
            )

        try:
            with ThreadPoolExecutor(max_workers=5) as executor:
                runs = list(executor.map(create, range(5)))
        finally:
            for catalog in catalogs:
                catalog.close()
        self.assertEqual(
            sorted(run.attempt_number for run in runs),
            [1, 2, 3, 4, 5],
        )
        self.assertEqual(len({run.thread_id for run in runs}), 5)

    def test_ownership_and_url_relationships_are_enforced(self):
        run = self.create_run()
        with self.assertRaises(ResearchProjectNotFoundError):
            self.service.get_run(
                self.project.project_id,
                self.task.task_id,
                run.run_id,
                "user_002",
            )
        with self.assertRaises(AgentRunNotFoundError):
            self.service.get_run(
                self.project.project_id,
                self.second_task.task_id,
                run.run_id,
                "user_001",
            )
        with self.assertRaises(ResearchTaskNotFoundError):
            self.service.get_run(
                self.other_project.project_id,
                self.task.task_id,
                run.run_id,
                "user_001",
            )

    def test_all_legal_transitions_and_timestamp_semantics(self):
        running = self.transition(self.create_run(), "running")
        self.assertIsNotNone(running.started_at)
        self.assertIsNone(running.finished_at)
        interrupted = self.transition(running, "interrupted")
        self.assertIsNone(interrupted.finished_at)
        resumed = self.transition(interrupted, "running")
        self.assertEqual(resumed.started_at, running.started_at)
        self.assertIsNone(resumed.finished_at)
        cancelled = self.transition(resumed, "cancelled")
        self.assertIsNotNone(cancelled.finished_at)

        pending_failed = self.transition(
            self.create_run(),
            "failed",
            error_message="准备失败",
        )
        self.assertIsNone(pending_failed.started_at)
        self.assertIsNotNone(pending_failed.finished_at)
        self.assertEqual(pending_failed.error_message, "准备失败")
        self.assertIsNotNone(self.transition(self.create_run(), "cancelled").finished_at)

        running_failed = self.transition(
            self.transition(self.create_run(), "running"),
            "failed",
            error_message="执行失败",
        )
        self.assertIsNotNone(running_failed.finished_at)
        self.assertIsNotNone(
            self.transition(
                self.transition(self.create_run(), "running"),
                "cancelled",
            ).finished_at
        )
        interrupted_failed = self.transition(
            self.transition(
                self.transition(self.create_run(), "running"),
                "interrupted",
            ),
            "failed",
            error_message="恢复失败",
        )
        self.assertIsNotNone(interrupted_failed.finished_at)
        interrupted_cancelled = self.transition(
            self.transition(
                self.transition(self.create_run(), "running"),
                "interrupted",
            ),
            "cancelled",
        )
        self.assertIsNotNone(interrupted_cancelled.finished_at)

    def test_failure_requires_sanitized_error_and_terminal_cannot_resume(self):
        run = self.create_run()
        with self.assertRaises(AgentRunValidationError):
            self.transition(run, "failed")
        failed = self.transition(
            run,
            "failed",
            error_message="API_KEY=secret-value\nToken=abc123 sk-test-123456",
        )
        self.assertNotIn("secret-value", failed.error_message)
        self.assertNotIn("abc123", failed.error_message)
        self.assertNotIn("sk-test", failed.error_message)
        self.assertIn("[REDACTED]", failed.error_message)
        with self.assertRaises(AgentRunConflictError):
            self.transition(failed, "running")
        cancelled = self.transition(self.create_run(), "cancelled")
        with self.assertRaises(AgentRunConflictError):
            self.transition(cancelled, "running")

    def test_completion_requires_valid_final_artifact_and_attachment_is_idempotent(self):
        running = self.transition(self.create_run(), "running")
        with self.assertRaises(AgentRunValidationError):
            self.transition(running, "completed")

        task_artifact = self.finalized_artifact(task_id=self.task.task_id)
        attached = self.service.attach_final_artifact(
            self.project.project_id,
            self.task.task_id,
            running.run_id,
            task_artifact.artifact_id,
            "user_001",
        )
        repeated = self.service.attach_final_artifact(
            self.project.project_id,
            self.task.task_id,
            running.run_id,
            task_artifact.artifact_id,
            "user_001",
        )
        self.assertEqual(repeated, attached)
        completed = self.transition(attached, "completed")
        self.assertEqual(completed.final_artifact_id, task_artifact.artifact_id)
        self.assertIsNotNone(completed.finished_at)
        self.assertEqual(
            self.service.attach_final_artifact(
                self.project.project_id,
                self.task.task_id,
                completed.run_id,
                task_artifact.artifact_id,
                "user_001",
            ),
            completed,
        )
        self.assertEqual(
            self.service.get_task(
                self.project.project_id,
                self.task.task_id,
                "user_001",
            ).status,
            "pending",
        )
        with self.assertRaises(AgentRunConflictError):
            self.transition(completed, "running")

        project_artifact = self.finalized_artifact(task_id=None)
        second = self.transition(self.create_run(), "running")
        project_attached = self.service.attach_final_artifact(
            self.project.project_id,
            self.task.task_id,
            second.run_id,
            project_artifact.artifact_id,
            "user_001",
        )
        self.assertEqual(project_attached.final_artifact_id, project_artifact.artifact_id)

    def test_invalid_artifact_bindings_and_rebinding_are_rejected(self):
        running = self.transition(self.create_run(), "running")
        draft = self.service.create_artifact(
            self.project.project_id,
            "user_001",
            task_id=self.task.task_id,
            title="草稿",
            content="草稿正文。",
        )
        wrong_task = self.finalized_artifact(task_id=self.second_task.task_id)
        other_project = self.finalized_artifact(
            task_id=self.other_task.task_id,
            project=self.other_project,
        )
        for artifact in (draft, wrong_task, other_project):
            with self.subTest(artifact=artifact.artifact_id):
                with self.assertRaises(
                    (
                        AgentRunConflictError,
                        ResearchArtifactConflictError,
                        ResearchArtifactNotFoundError,
                    )
                ):
                    self.service.attach_final_artifact(
                        self.project.project_id,
                        self.task.task_id,
                        running.run_id,
                        artifact.artifact_id,
                        "user_001",
                    )

        first = self.finalized_artifact(task_id=self.task.task_id)
        second = self.finalized_artifact(task_id=self.task.task_id)
        self.service.attach_final_artifact(
            self.project.project_id,
            self.task.task_id,
            running.run_id,
            first.artifact_id,
            "user_001",
        )
        with self.assertRaises(AgentRunConflictError):
            self.service.attach_final_artifact(
                self.project.project_id,
                self.task.task_id,
                running.run_id,
                second.artifact_id,
                "user_001",
            )

    def test_runs_restrict_task_project_and_artifact_deletion(self):
        run = self.transition(self.create_run(), "running")
        artifact = self.finalized_artifact(task_id=self.task.task_id)
        self.service.attach_final_artifact(
            self.project.project_id,
            self.task.task_id,
            run.run_id,
            artifact.artifact_id,
            "user_001",
        )
        with self.assertRaises(ResearchTaskConflictError):
            self.service.delete_task(
                self.project.project_id,
                self.task.task_id,
                "user_001",
            )
        with self.assertRaises(ResearchProjectConflictError):
            self.service.delete_project(self.project.project_id, "user_001")
        with self.assertRaises(ResearchArtifactConflictError):
            self.service.delete_artifact(
                self.project.project_id,
                artifact.artifact_id,
                "user_001",
            )

    def test_run_survives_service_restart(self):
        run = self.create_run()
        self.catalog.close()
        self.catalog = ResearchCatalog(self.path)
        self.service = ResearchService(self.catalog, KnowledgeStub())
        restored = self.service.get_run(
            self.project.project_id,
            self.task.task_id,
            run.run_id,
            "user_001",
        )
        self.assertEqual(restored, run)


if __name__ == "__main__":
    unittest.main()
