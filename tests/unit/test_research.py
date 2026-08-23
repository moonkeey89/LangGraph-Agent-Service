import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ai_agent_learning.knowledge.service import KnowledgeNotFoundError
from ai_agent_learning.research import (
    CURRENT_SCHEMA_VERSION,
    ResearchCatalog,
    ResearchKnowledgeBaseNotFoundError,
    ResearchProjectNotFoundError,
    ResearchProjectValidationError,
    ResearchService,
)


class KnowledgeOwnershipStub:
    def __init__(self):
        self.owners = {
            "kb_user_001": "user_001",
            "kb_user_002": "user_002",
        }

    def ensure_owned(self, knowledge_base_id: str, owner_user_id: str) -> None:
        if self.owners.get(knowledge_base_id) != owner_user_id:
            raise KnowledgeNotFoundError


class ResearchCatalogTests(unittest.TestCase):
    def test_first_initialization_creates_schema_and_migration(self):
        with TemporaryDirectory() as temporary:
            catalog = ResearchCatalog(Path(temporary) / "research.sqlite")
            try:
                tables = {
                    row[0]
                    for row in catalog.connection.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    ).fetchall()
                }
                versions = catalog.connection.execute(
                    "SELECT version FROM schema_migrations"
                ).fetchall()
                self.assertIn("schema_migrations", tables)
                self.assertIn("research_projects", tables)
                self.assertEqual(
                    [row[0] for row in versions],
                    [CURRENT_SCHEMA_VERSION],
                )
            finally:
                catalog.close()

    def test_repeated_initialization_is_idempotent_and_preserves_data(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "research.sqlite"
            knowledge = KnowledgeOwnershipStub()
            first = ResearchCatalog(path)
            project = ResearchService(first, knowledge).create_project(
                owner_user_id="user_001",
                name="复现实验",
            )
            first.close()

            second = ResearchCatalog(path)
            try:
                restored = second.get_by_id(project.project_id)
                versions = second.connection.execute(
                    "SELECT version FROM schema_migrations"
                ).fetchall()
                self.assertIsNotNone(restored)
                self.assertEqual(restored.name, "复现实验")
                self.assertEqual(len(versions), 1)
            finally:
                second.close()


class ResearchServiceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.path = Path(self.temporary.name) / "research.sqlite"
        self.catalog = ResearchCatalog(self.path)
        self.knowledge = KnowledgeOwnershipStub()
        self.service = ResearchService(self.catalog, self.knowledge)

    def tearDown(self):
        self.catalog.close()
        self.temporary.cleanup()

    def create_project(self, user_id: str = "user_001"):
        return self.service.create_project(
            owner_user_id=user_id,
            name="  量子材料综述  ",
            description="  项目说明  ",
            research_question="  哪些方法最可靠？  ",
        )

    def test_create_generates_identity_normalizes_fields_and_lists_by_owner(self):
        project = self.create_project()
        other = self.create_project("user_002")

        self.assertTrue(project.project_id.startswith("rp_"))
        self.assertEqual(project.owner_user_id, "user_001")
        self.assertEqual(project.name, "量子材料综述")
        self.assertEqual(project.description, "项目说明")
        self.assertEqual(project.research_question, "哪些方法最可靠？")
        self.assertEqual(project.status, "draft")
        self.assertEqual(
            [item.project_id for item in self.service.list_projects("user_001")],
            [project.project_id],
        )
        self.assertNotEqual(project.project_id, other.project_id)

    def test_foreign_project_is_hidden_for_get_update_and_delete(self):
        project = self.create_project("user_002")

        operations = (
            lambda: self.service.get_project(project.project_id, "user_001"),
            lambda: self.service.update_project(
                project.project_id,
                "user_001",
                name="越权修改",
            ),
            lambda: self.service.delete_project(
                project.project_id,
                "user_001",
            ),
        )
        for operation in operations:
            with self.subTest(operation=operation):
                with self.assertRaises(ResearchProjectNotFoundError):
                    operation()

        self.assertEqual(
            self.service.get_project(project.project_id, "user_002").name,
            "量子材料综述",
        )

    def test_update_all_fields_and_partial_update_preserves_omitted_fields(self):
        project = self.create_project()
        updated = self.service.update_project(
            project.project_id,
            "user_001",
            name="新名称",
            description="新说明",
            research_question="新问题",
            status="active",
        )
        partial = self.service.update_project(
            project.project_id,
            "user_001",
            name="最终名称",
        )

        self.assertEqual(updated.status, "active")
        self.assertEqual(partial.name, "最终名称")
        self.assertEqual(partial.description, "新说明")
        self.assertEqual(partial.research_question, "新问题")
        self.assertEqual(partial.status, "active")
        self.assertEqual(partial.created_at, project.created_at)

    def test_invalid_name_status_and_empty_patch_are_rejected(self):
        project = self.create_project()
        invalid_operations = (
            lambda: self.service.create_project(
                owner_user_id="user_001",
                name="   ",
            ),
            lambda: self.service.update_project(
                project.project_id,
                "user_001",
                status="unknown",
            ),
            lambda: self.service.update_project(
                project.project_id,
                "user_001",
            ),
        )
        for operation in invalid_operations:
            with self.subTest(operation=operation):
                with self.assertRaises(ResearchProjectValidationError):
                    operation()

    def test_owned_knowledge_base_can_be_bound_and_unbound(self):
        project = self.service.create_project(
            owner_user_id="user_001",
            name="知识库项目",
            default_knowledge_base_id="kb_user_001",
        )
        self.assertEqual(
            project.default_knowledge_base_id,
            "kb_user_001",
        )

        updated = self.service.update_project(
            project.project_id,
            "user_001",
            default_knowledge_base_id=None,
        )
        self.assertIsNone(updated.default_knowledge_base_id)

    def test_foreign_knowledge_base_cannot_be_bound(self):
        with self.assertRaises(ResearchKnowledgeBaseNotFoundError):
            self.service.create_project(
                owner_user_id="user_001",
                name="越权绑定",
                default_knowledge_base_id="kb_user_002",
            )

        project = self.create_project()
        with self.assertRaises(ResearchKnowledgeBaseNotFoundError):
            self.service.update_project(
                project.project_id,
                "user_001",
                default_knowledge_base_id="kb_user_002",
            )

    def test_delete_removes_only_project_and_not_knowledge_base(self):
        project = self.service.create_project(
            owner_user_id="user_001",
            name="待删除项目",
            default_knowledge_base_id="kb_user_001",
        )
        self.service.delete_project(project.project_id, "user_001")

        with self.assertRaises(ResearchProjectNotFoundError):
            self.service.get_project(project.project_id, "user_001")
        self.assertEqual(self.knowledge.owners["kb_user_001"], "user_001")

    def test_project_survives_catalog_restart(self):
        project = self.create_project()
        self.catalog.close()
        self.catalog = ResearchCatalog(self.path)
        self.service = ResearchService(self.catalog, self.knowledge)

        restored = self.service.get_project(project.project_id, "user_001")
        self.assertEqual(restored, project)


if __name__ == "__main__":
    unittest.main()
