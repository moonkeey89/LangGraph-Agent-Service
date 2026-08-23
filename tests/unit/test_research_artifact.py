import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ai_agent_learning.knowledge import (
    ChromaKnowledgeRepository,
    KnowledgeCatalog,
    KnowledgeIngestor,
    KnowledgeLibraryService,
)
from ai_agent_learning.knowledge.models import KnowledgeChunk
from ai_agent_learning.research import (
    CURRENT_SCHEMA_VERSION,
    ResearchArtifact,
    ResearchArtifactConflictError,
    ResearchArtifactNotFoundError,
    ResearchArtifactSourceNotFoundError,
    ResearchCatalog,
    ResearchProjectConflictError,
    ResearchProjectNotFoundError,
    ResearchService,
    ResearchTaskConflictError,
    ResearchTaskNotFoundError,
)
from tests.helpers import DeterministicTestEmbeddings


def _create_version_two_database(path: Path) -> tuple[str, str]:
    project_id = "rp_existing"
    task_id = "rt_existing"
    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            """
            PRAGMA foreign_keys = ON;
            CREATE TABLE schema_migrations (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL
            );
            INSERT INTO schema_migrations VALUES (1, '2026-01-01T00:00:00+00:00');
            INSERT INTO schema_migrations VALUES (2, '2026-01-02T00:00:00+00:00');
            CREATE TABLE research_projects (
                project_id TEXT PRIMARY KEY,
                owner_user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                description TEXT NOT NULL,
                research_question TEXT NOT NULL,
                status TEXT NOT NULL,
                default_knowledge_base_id TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE research_tasks (
                task_id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                title TEXT NOT NULL,
                objective TEXT NOT NULL,
                task_type TEXT NOT NULL,
                status TEXT NOT NULL,
                acceptance_criteria TEXT NOT NULL,
                result_summary TEXT,
                error_message TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                FOREIGN KEY (project_id) REFERENCES research_projects(project_id)
                    ON DELETE RESTRICT
            );
            """
        )
        timestamp = "2026-01-01T00:00:00+00:00"
        connection.execute(
            "INSERT INTO research_projects VALUES (?, ?, ?, '', '', 'active', NULL, ?, ?)",
            (project_id, "user_001", "已有项目", timestamp, timestamp),
        )
        connection.execute(
            "INSERT INTO research_tasks VALUES (?, ?, ?, '', 'general', 'pending', '[]', NULL, NULL, ?, ?, NULL, NULL)",
            (task_id, project_id, "已有任务", timestamp, timestamp),
        )
        connection.commit()
    finally:
        connection.close()
    return project_id, task_id


class ResearchArtifactMigrationTests(unittest.TestCase):
    def test_version_two_upgrades_to_three_and_preserves_records(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "research.sqlite"
            project_id, task_id = _create_version_two_database(path)
            catalog = ResearchCatalog(path)
            try:
                versions = [
                    row[0]
                    for row in catalog.connection.execute(
                        "SELECT version FROM schema_migrations ORDER BY version"
                    )
                ]
                tables = {
                    row[0]
                    for row in catalog.connection.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                }
                self.assertEqual(versions, [1, 2, CURRENT_SCHEMA_VERSION])
                self.assertIn("research_artifacts", tables)
                self.assertEqual(catalog.get_by_id(project_id).name, "已有项目")
                self.assertEqual(catalog.get_task(project_id, task_id).title, "已有任务")
            finally:
                catalog.close()

            restarted = ResearchCatalog(path)
            try:
                count = restarted.connection.execute(
                    "SELECT COUNT(*) FROM schema_migrations"
                ).fetchone()[0]
                self.assertEqual(count, 3)
                self.assertEqual(
                    restarted.connection.execute("PRAGMA foreign_keys").fetchone()[0],
                    1,
                )
            finally:
                restarted.close()

    def test_artifact_foreign_keys_are_actually_enforced(self):
        with TemporaryDirectory() as temporary:
            catalog = ResearchCatalog(Path(temporary) / "research.sqlite")
            try:
                artifact = ResearchArtifact(
                    artifact_id="ra_invalid",
                    project_id="rp_missing",
                    task_id=None,
                    title="非法成果",
                    artifact_type="note",
                    content="不能写入。",
                    status="draft",
                    created_by="user",
                    sources=[],
                    created_at="2026-01-01T00:00:00+00:00",
                    updated_at="2026-01-01T00:00:00+00:00",
                    finalized_at=None,
                )
                with self.assertRaises(sqlite3.IntegrityError):
                    catalog.create_artifact(artifact)
            finally:
                catalog.close()


class ResearchArtifactServiceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.research_catalog = ResearchCatalog(self.root / "research.sqlite")
        self.knowledge_catalog = KnowledgeCatalog(self.root / "knowledge.sqlite")
        self.repository = ChromaKnowledgeRepository(
            persist_directory=self.root / "chroma",
            embeddings=DeterministicTestEmbeddings(),
        )
        self.ingestor = KnowledgeIngestor(
            self.repository,
            chunk_size=80,
            chunk_overlap=10,
        )
        self.knowledge_service = KnowledgeLibraryService(
            catalog=self.knowledge_catalog,
            repository=self.repository,
            ingestor=self.ingestor,
            source_directory=self.root / "sources",
            max_file_size_bytes=1024 * 1024,
            max_files_per_upload=5,
        )
        self.service = ResearchService(
            self.research_catalog,
            self.knowledge_service,
        )
        self.user_base = self.knowledge_service.create_knowledge_base(
            owner_user_id="user_001",
            name="用户一知识库",
        )
        self.other_base = self.knowledge_service.create_knowledge_base(
            owner_user_id="user_002",
            name="用户二知识库",
        )
        self.ready_chunk = self.seed_chunk(
            self.user_base.knowledge_base_id,
            "doc-ready",
            "chunk-ready",
            "研究表明模型A在测试集上达到91%。",
            status="ready",
        )
        self.processing_chunk = self.seed_chunk(
            self.user_base.knowledge_base_id,
            "doc-processing",
            "chunk-processing",
            "这段内容尚未完成索引。",
            status="processing",
        )
        self.other_chunk = self.seed_chunk(
            self.other_base.knowledge_base_id,
            "doc-other",
            "chunk-other",
            "其他用户的私有证据。",
            status="ready",
        )
        self.project = self.service.create_project(
            owner_user_id="user_001",
            name="科研项目",
            status="active",
            default_knowledge_base_id=self.user_base.knowledge_base_id,
        )
        self.other_project = self.service.create_project(
            owner_user_id="user_001",
            name="其他项目",
            status="active",
        )
        self.task = self.service.create_task(
            self.project.project_id,
            "user_001",
            title="分析实验结果",
        )
        self.other_task = self.service.create_task(
            self.other_project.project_id,
            "user_001",
            title="其他项目任务",
        )

    def tearDown(self):
        self.repository.close()
        self.knowledge_catalog.close()
        self.research_catalog.close()
        self.temporary.cleanup()

    def seed_chunk(
        self,
        knowledge_base_id: str,
        document_id: str,
        chunk_id: str,
        content: str,
        *,
        status: str,
    ) -> KnowledgeChunk:
        self.knowledge_catalog.save_processing_document(
            document_id=document_id,
            knowledge_base_id=knowledge_base_id,
            original_filename=f"{document_id}.md",
            stored_filename=f"{knowledge_base_id}/{document_id}.md",
            content_hash=f"hash-{document_id}",
            content_type="text/markdown",
            size=len(content.encode()),
        )
        chunk = KnowledgeChunk(
            content=content,
            knowledge_base_id=knowledge_base_id,
            document_id=document_id,
            source=f"{document_id}.md",
            page=2,
            chunk_id=chunk_id,
        )
        self.repository.replace_document([chunk])
        if status == "ready":
            self.knowledge_catalog.mark_document_ready(document_id, 1)
        return chunk

    def create_artifact(self, **overrides):
        values = {
            "title": "实验分析草稿",
            "content": "模型A的测试结果需要进一步讨论。",
            "artifact_type": "analysis",
        }
        values.update(overrides)
        return self.service.create_artifact(
            self.project.project_id,
            "user_001",
            **values,
        )

    def test_create_project_and_task_artifacts_and_internal_agent_creator(self):
        project_artifact = self.create_artifact()
        task_artifact = self.create_artifact(task_id=self.task.task_id)
        agent_artifact = self.create_artifact(created_by="agent")
        self.assertIsNone(project_artifact.task_id)
        self.assertEqual(project_artifact.status, "draft")
        self.assertEqual(project_artifact.created_by, "user")
        self.assertEqual(task_artifact.task_id, self.task.task_id)
        self.assertEqual(agent_artifact.created_by, "agent")

        with self.assertRaises(ResearchTaskNotFoundError):
            self.create_artifact(task_id=self.other_task.task_id)

    def test_sources_are_resolved_from_ready_owned_chunk_and_persisted(self):
        artifact = self.create_artifact(
            source_chunk_ids=[self.ready_chunk.chunk_id, self.ready_chunk.chunk_id]
        )
        self.assertEqual(len(artifact.sources), 1)
        source = artifact.sources[0]
        self.assertEqual(source.knowledge_base_id, self.user_base.knowledge_base_id)
        self.assertEqual(source.document_id, self.ready_chunk.document_id)
        self.assertEqual(source.source, "doc-ready.md")
        self.assertEqual(source.page, 2)
        self.assertEqual(source.excerpt, self.ready_chunk.content)

        restored = self.service.get_artifact(
            self.project.project_id,
            artifact.artifact_id,
            "user_001",
        )
        self.assertEqual(restored.sources, artifact.sources)

    def test_missing_processing_and_other_user_chunks_are_rejected(self):
        for chunk_id in (
            "chunk-missing",
            self.processing_chunk.chunk_id,
            self.other_chunk.chunk_id,
        ):
            with self.subTest(chunk_id=chunk_id):
                with self.assertRaises(ResearchArtifactSourceNotFoundError):
                    self.create_artifact(source_chunk_ids=[chunk_id])

    def test_draft_update_finalize_and_terminal_rules(self):
        artifact = self.create_artifact(source_chunk_ids=[self.ready_chunk.chunk_id])
        updated = self.service.update_artifact(
            self.project.project_id,
            artifact.artifact_id,
            "user_001",
            title="更新后的分析",
            content="更新后的正文。",
            artifact_type="report",
            source_chunk_ids=[],
        )
        self.assertEqual(updated.title, "更新后的分析")
        self.assertEqual(updated.sources, [])
        finalized = self.service.finalize_artifact(
            self.project.project_id,
            artifact.artifact_id,
            "user_001",
        )
        self.assertEqual(finalized.status, "final")
        self.assertIsNotNone(finalized.finalized_at)
        self.assertEqual(finalized.updated_at, finalized.finalized_at)
        with self.assertRaises(ResearchArtifactConflictError):
            self.service.finalize_artifact(
                self.project.project_id,
                artifact.artifact_id,
                "user_001",
            )
        with self.assertRaises(ResearchArtifactConflictError):
            self.service.update_artifact(
                self.project.project_id,
                artifact.artifact_id,
                "user_001",
                title="不可修改",
            )
        with self.assertRaises(ResearchArtifactConflictError):
            self.service.delete_artifact(
                self.project.project_id,
                artifact.artifact_id,
                "user_001",
            )

    def test_draft_delete_preserves_task_and_knowledge(self):
        artifact = self.create_artifact(
            task_id=self.task.task_id,
            source_chunk_ids=[self.ready_chunk.chunk_id],
        )
        self.service.delete_artifact(
            self.project.project_id,
            artifact.artifact_id,
            "user_001",
        )
        with self.assertRaises(ResearchArtifactNotFoundError):
            self.service.get_artifact(
                self.project.project_id,
                artifact.artifact_id,
                "user_001",
            )
        self.assertEqual(
            self.service.get_task(
                self.project.project_id,
                self.task.task_id,
                "user_001",
            ).task_id,
            self.task.task_id,
        )
        self.assertIsNotNone(
            self.repository.get_chunk(
                knowledge_base_id=self.user_base.knowledge_base_id,
                chunk_id=self.ready_chunk.chunk_id,
            )
        )

    def test_artifacts_restrict_project_and_task_deletion(self):
        project_artifact = self.create_artifact()
        with self.assertRaises(ResearchProjectConflictError):
            self.service.delete_project(self.project.project_id, "user_001")
        self.service.delete_artifact(
            self.project.project_id,
            project_artifact.artifact_id,
            "user_001",
        )
        task_artifact = self.create_artifact(task_id=self.task.task_id)
        with self.assertRaises(ResearchTaskConflictError):
            self.service.delete_task(
                self.project.project_id,
                self.task.task_id,
                "user_001",
            )
        self.service.delete_artifact(
            self.project.project_id,
            task_artifact.artifact_id,
            "user_001",
        )

    def test_archived_project_artifacts_are_read_only(self):
        artifact = self.create_artifact()
        self.service.update_project(
            self.project.project_id,
            "user_001",
            status="archived",
        )
        self.assertEqual(
            self.service.get_artifact(
                self.project.project_id,
                artifact.artifact_id,
                "user_001",
            ).artifact_id,
            artifact.artifact_id,
        )
        operations = (
            lambda: self.create_artifact(),
            lambda: self.service.update_artifact(
                self.project.project_id,
                artifact.artifact_id,
                "user_001",
                title="不可修改",
            ),
            lambda: self.service.finalize_artifact(
                self.project.project_id,
                artifact.artifact_id,
                "user_001",
            ),
            lambda: self.service.delete_artifact(
                self.project.project_id,
                artifact.artifact_id,
                "user_001",
            ),
        )
        for operation in operations:
            with self.subTest(operation=operation):
                with self.assertRaises(ResearchArtifactConflictError):
                    operation()

    def test_filters_ownership_membership_and_restart(self):
        first = self.create_artifact(task_id=self.task.task_id)
        self.create_artifact(artifact_type="report")
        self.assertEqual(
            self.service.list_artifacts(
                self.project.project_id,
                "user_001",
                task_id=self.task.task_id,
            ),
            [first],
        )
        with self.assertRaises(ResearchTaskNotFoundError):
            self.service.list_artifacts(
                self.project.project_id,
                "user_001",
                task_id=self.other_task.task_id,
            )
        with self.assertRaises(ResearchProjectNotFoundError):
            self.service.get_artifact(
                self.project.project_id,
                first.artifact_id,
                "user_002",
            )

        self.research_catalog.close()
        self.research_catalog = ResearchCatalog(self.root / "research.sqlite")
        self.service = ResearchService(
            self.research_catalog,
            self.knowledge_service,
        )
        restored = self.service.get_artifact(
            self.project.project_id,
            first.artifact_id,
            "user_001",
        )
        self.assertEqual(restored, first)


if __name__ == "__main__":
    unittest.main()
