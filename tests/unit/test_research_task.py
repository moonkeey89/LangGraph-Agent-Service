import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ai_agent_learning.knowledge.service import KnowledgeNotFoundError
from ai_agent_learning.research import (
    CURRENT_SCHEMA_VERSION,
    ResearchCatalog,
    ResearchProjectConflictError,
    ResearchProjectNotFoundError,
    ResearchTask,
    ResearchTaskConflictError,
    ResearchTaskNotFoundError,
    ResearchTaskValidationError,
    ResearchService,
)


class KnowledgeOwnershipStub:
    def ensure_owned(self, _knowledge_base_id: str, _owner_user_id: str) -> None:
        raise KnowledgeNotFoundError


def _create_version_one_database(path: Path) -> str:
    project_id = "rp_existing"
    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            """
            CREATE TABLE schema_migrations (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL
            );
            INSERT INTO schema_migrations VALUES (1, '2026-01-01T00:00:00+00:00');
            CREATE TABLE research_projects (
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
            """
        )
        connection.execute(
            """
            INSERT INTO research_projects VALUES (
                ?, 'user_001', '已有项目', '', '', 'draft', NULL,
                '2026-01-01T00:00:00+00:00',
                '2026-01-01T00:00:00+00:00'
            )
            """,
            (project_id,),
        )
        connection.commit()
    finally:
        connection.close()
    return project_id


class ResearchTaskMigrationTests(unittest.TestCase):
    def test_version_one_upgrades_to_two_and_preserves_project(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "research.sqlite"
            project_id = _create_version_one_database(path)
            catalog = ResearchCatalog(path)
            try:
                tables = {
                    row[0]
                    for row in catalog.connection.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    ).fetchall()
                }
                versions = [
                    row[0]
                    for row in catalog.connection.execute(
                        "SELECT version FROM schema_migrations ORDER BY version"
                    ).fetchall()
                ]
                self.assertIn("research_tasks", tables)
                self.assertEqual(versions, [1, 2, 3, CURRENT_SCHEMA_VERSION])
                self.assertEqual(catalog.get_by_id(project_id).name, "已有项目")
            finally:
                catalog.close()

    def test_version_two_reinitialization_is_idempotent(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "research.sqlite"
            project_id = _create_version_one_database(path)
            first = ResearchCatalog(path)
            service = ResearchService(first, KnowledgeOwnershipStub())
            task = service.create_task(
                project_id,
                "user_001",
                title="持久化任务",
            )
            first.close()

            second = ResearchCatalog(path)
            try:
                restored = second.get_task(project_id, task.task_id)
                count = second.connection.execute(
                    "SELECT COUNT(*) FROM schema_migrations"
                ).fetchone()[0]
                self.assertEqual(restored.title, "持久化任务")
                self.assertEqual(count, 4)
            finally:
                second.close()

    def test_foreign_key_is_enabled_and_restricts_invalid_project(self):
        with TemporaryDirectory() as temporary:
            catalog = ResearchCatalog(Path(temporary) / "research.sqlite")
            try:
                enabled = catalog.connection.execute(
                    "PRAGMA foreign_keys"
                ).fetchone()[0]
                self.assertEqual(enabled, 1)
                invalid = ResearchTask(
                    task_id="rt_invalid",
                    project_id="rp_missing",
                    title="非法任务",
                    objective="",
                    task_type="general",
                    status="pending",
                    acceptance_criteria=[],
                    result_summary=None,
                    error_message=None,
                    created_at="2026-01-01T00:00:00+00:00",
                    updated_at="2026-01-01T00:00:00+00:00",
                    started_at=None,
                    completed_at=None,
                )
                with self.assertRaises(sqlite3.IntegrityError):
                    catalog.create_task(invalid)

                service = ResearchService(catalog, KnowledgeOwnershipStub())
                project = service.create_project(
                    owner_user_id="user_001",
                    name="外键删除测试",
                )
                service.create_task(
                    project.project_id,
                    "user_001",
                    title="受保护任务",
                )
                with self.assertRaises(sqlite3.IntegrityError):
                    catalog.delete(project.project_id, "user_001")
            finally:
                catalog.close()


class ResearchTaskServiceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.path = Path(self.temporary.name) / "research.sqlite"
        self.catalog = ResearchCatalog(self.path)
        self.service = ResearchService(self.catalog, KnowledgeOwnershipStub())
        self.project = self.service.create_project(
            owner_user_id="user_001",
            name="科研项目",
        )
        self.other_project = self.service.create_project(
            owner_user_id="user_001",
            name="另一个项目",
        )

    def tearDown(self):
        self.catalog.close()
        self.temporary.cleanup()

    def create_task(self, project_id: str | None = None):
        return self.service.create_task(
            project_id or self.project.project_id,
            "user_001",
            title="  文献回顾  ",
            objective="  汇总研究方法  ",
            task_type="literature_review",
            acceptance_criteria=[" 至少引用三篇论文 ", "列出方法差异"],
        )

    def transition(self, task, target_status, **kwargs):
        return self.service.transition_task(
            task.project_id,
            task.task_id,
            "user_001",
            target_status=target_status,
            **kwargs,
        )

    def test_create_defaults_and_acceptance_criteria_round_trip(self):
        task = self.create_task()
        restored = self.service.get_task(
            self.project.project_id,
            task.task_id,
            "user_001",
        )
        self.assertTrue(task.task_id.startswith("rt_"))
        self.assertEqual(task.project_id, self.project.project_id)
        self.assertEqual(task.title, "文献回顾")
        self.assertEqual(task.objective, "汇总研究方法")
        self.assertEqual(task.status, "pending")
        self.assertEqual(
            restored.acceptance_criteria,
            ["至少引用三篇论文", "列出方法差异"],
        )
        self.assertIsNone(task.result_summary)
        self.assertIsNone(task.error_message)
        self.assertIsNone(task.started_at)
        self.assertIsNone(task.completed_at)

    def test_owner_and_project_membership_are_both_required(self):
        task = self.create_task()
        operations = (
            lambda: self.service.get_task(
                self.project.project_id,
                task.task_id,
                "user_002",
            ),
            lambda: self.service.update_task(
                self.project.project_id,
                task.task_id,
                "user_002",
                title="越权",
            ),
            lambda: self.service.transition_task(
                self.project.project_id,
                task.task_id,
                "user_002",
                target_status="running",
            ),
            lambda: self.service.delete_task(
                self.project.project_id,
                task.task_id,
                "user_002",
            ),
        )
        for operation in operations:
            with self.subTest(operation=operation):
                with self.assertRaises(ResearchProjectNotFoundError):
                    operation()

        with self.assertRaises(ResearchTaskNotFoundError):
            self.service.get_task(
                self.other_project.project_id,
                task.task_id,
                "user_001",
            )

    def test_normal_update_changes_only_editable_fields(self):
        task = self.create_task()
        updated = self.service.update_task(
            task.project_id,
            task.task_id,
            "user_001",
            title="新标题",
            objective="新目标",
            task_type="analysis",
            acceptance_criteria=["输出计算结果"],
        )
        partial = self.service.update_task(
            task.project_id,
            task.task_id,
            "user_001",
            title="最终标题",
        )
        self.assertEqual(updated.task_type, "analysis")
        self.assertEqual(partial.title, "最终标题")
        self.assertEqual(partial.objective, "新目标")
        self.assertEqual(partial.status, "pending")
        self.assertIsNone(partial.started_at)

    def test_invalid_task_fields_are_rejected(self):
        invalid_operations = (
            lambda: self.service.create_task(
                self.project.project_id,
                "user_001",
                title="  ",
            ),
            lambda: self.service.create_task(
                self.project.project_id,
                "user_001",
                title="任务",
                task_type="unknown",
            ),
            lambda: self.service.create_task(
                self.project.project_id,
                "user_001",
                title="任务",
                acceptance_criteria=["   "],
            ),
        )
        for operation in invalid_operations:
            with self.subTest(operation=operation):
                with self.assertRaises(ResearchTaskValidationError):
                    operation()

    def test_all_legal_transitions_and_timestamps(self):
        running = self.transition(self.create_task(), "running")
        self.assertIsNotNone(running.started_at)
        self.assertIsNone(running.completed_at)

        blocked = self.transition(running, "blocked", reason="等待数据")
        self.assertEqual(blocked.error_message, "等待数据")
        resumed = self.transition(blocked, "running")
        self.assertEqual(resumed.started_at, running.started_at)
        self.assertIsNone(resumed.error_message)
        cancelled_after_block = self.transition(resumed, "cancelled")
        self.assertIsNotNone(cancelled_after_block.completed_at)

        blocked_cancelled = self.transition(
            self.transition(
                self.transition(self.create_task(), "running"),
                "blocked",
                reason="等待审批",
            ),
            "cancelled",
        )
        self.assertEqual(blocked_cancelled.status, "cancelled")

        completed = self.transition(
            self.transition(self.create_task(), "running"),
            "completed",
            result_summary="完成综述",
        )
        self.assertEqual(completed.result_summary, "完成综述")
        self.assertIsNotNone(completed.completed_at)
        self.assertIsNone(completed.error_message)

        failed = self.transition(
            self.transition(self.create_task(), "running"),
            "failed",
            reason="数据损坏",
        )
        self.assertEqual(failed.error_message, "数据损坏")
        self.assertIsNotNone(failed.completed_at)
        retried = self.transition(failed, "pending")
        self.assertIsNone(retried.error_message)
        self.assertIsNone(retried.completed_at)
        self.assertIsNone(retried.started_at)

        pending_cancelled = self.transition(self.create_task(), "cancelled")
        self.assertEqual(pending_cancelled.status, "cancelled")

    def test_illegal_transitions_and_missing_reason_are_rejected(self):
        pending = self.create_task()
        with self.assertRaises(ResearchTaskConflictError):
            self.transition(pending, "completed")
        running = self.transition(pending, "running")
        with self.assertRaises(ResearchTaskValidationError):
            self.transition(running, "failed")
        with self.assertRaises(ResearchTaskValidationError):
            self.transition(running, "blocked")
        completed = self.transition(running, "completed")
        with self.assertRaises(ResearchTaskConflictError):
            self.transition(completed, "running")

    def test_only_pending_and_cancelled_tasks_can_be_deleted(self):
        pending = self.create_task()
        self.service.delete_task(
            pending.project_id,
            pending.task_id,
            "user_001",
        )

        cancelled = self.transition(self.create_task(), "cancelled")
        self.service.delete_task(
            cancelled.project_id,
            cancelled.task_id,
            "user_001",
        )

        running = self.transition(self.create_task(), "running")
        blocked = self.transition(
            self.transition(self.create_task(), "running"),
            "blocked",
            reason="等待数据",
        )
        completed = self.transition(
            self.transition(self.create_task(), "running"),
            "completed",
        )
        failed = self.transition(
            self.transition(self.create_task(), "running"),
            "failed",
            reason="执行失败",
        )
        for protected in (running, blocked, completed, failed):
            with self.subTest(status=protected.status):
                with self.assertRaises(ResearchTaskConflictError):
                    self.service.delete_task(
                        protected.project_id,
                        protected.task_id,
                        "user_001",
                    )

    def test_project_with_task_cannot_be_deleted(self):
        self.create_task()
        with self.assertRaises(ResearchProjectConflictError):
            self.service.delete_project(self.project.project_id, "user_001")

    def test_archived_project_tasks_are_read_only(self):
        task = self.create_task()
        self.service.update_project(
            self.project.project_id,
            "user_001",
            status="archived",
        )
        self.assertEqual(
            self.service.get_task(
                self.project.project_id,
                task.task_id,
                "user_001",
            ).task_id,
            task.task_id,
        )
        operations = (
            lambda: self.create_task(),
            lambda: self.service.update_task(
                self.project.project_id,
                task.task_id,
                "user_001",
                title="不可修改",
            ),
            lambda: self.transition(task, "running"),
            lambda: self.service.delete_task(
                self.project.project_id,
                task.task_id,
                "user_001",
            ),
        )
        for operation in operations:
            with self.subTest(operation=operation):
                with self.assertRaises(ResearchTaskConflictError):
                    operation()

    def test_task_survives_service_restart(self):
        task = self.create_task()
        self.catalog.close()
        self.catalog = ResearchCatalog(self.path)
        self.service = ResearchService(self.catalog, KnowledgeOwnershipStub())
        restored = self.service.get_task(
            task.project_id,
            task.task_id,
            "user_001",
        )
        self.assertEqual(restored, task)


if __name__ == "__main__":
    unittest.main()
