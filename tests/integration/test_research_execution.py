import sqlite3
import unittest
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient

from ai_agent_learning.api.app import create_app
from ai_agent_learning.api.service import AgentService
from ai_agent_learning.knowledge.models import KnowledgeChunk
from ai_agent_learning.knowledge.service import KnowledgeNotFoundError
from ai_agent_learning.research import (
    CURRENT_SCHEMA_VERSION,
    AgentRunConflictError,
    ResearchCatalog,
    ResearchExecutionService,
    ResearchProjectNotFoundError,
    ResearchService,
    ResearchTaskConflictError,
    ResearchTaskNotFoundError,
)
from tests.helpers import install_test_identity


class KnowledgeStub:
    def __init__(self):
        self.chunks = {
            "chunk_001": KnowledgeChunk(
                content="真实证据：ResearchFlow 编号 RF-2026。",
                knowledge_base_id="kb_owned",
                document_id="doc_001",
                source="evidence.md",
                page=2,
                chunk_id="chunk_001",
            )
        }

    def ensure_owned(self, knowledge_base_id, owner_user_id):
        if knowledge_base_id != "kb_owned" or owner_user_id != "user_001":
            raise KnowledgeNotFoundError

    def get_ready_chunk(self, *, knowledge_base_id, owner_user_id, chunk_id):
        self.ensure_owned(knowledge_base_id, owner_user_id)
        chunk = self.chunks.get(chunk_id)
        if chunk is None:
            raise KnowledgeNotFoundError
        return chunk


class FakeResearchGraph:
    def __init__(self, result=None, *, error=None, catalog=None):
        self.result = result or {
            "outcome": "completed",
            "final_answer": "基于证据完成科研回答。",
            "sources": [],
            "unresolved_issues": [],
            "error": None,
        }
        self.error = error
        self.catalog = catalog
        self.calls = []

    def stream(
        self,
        state,
        *,
        config,
        context,
        stream_mode,
        subgraphs,
        version,
    ):
        if self.catalog is not None:
            assert not self.catalog.connection.in_transaction
        self.calls.append(
            {"state": dict(state), "config": config, "context": context}
        )
        assert stream_mode == ["updates", "messages"]
        assert subgraphs is False
        assert version == "v2"
        if self.error is not None:
            raise self.error
        yield {
            "type": "updates",
            "data": {"research_validate_binding": {}},
        }
        yield {
            "type": "updates",
            "data": {"research_supervisor": {"route": "direct"}},
        }
        if self.result.get("sources"):
            yield {
                "type": "updates",
                "data": {
                    "research_evidence_agent": {
                        "sources": list(self.result["sources"])
                    }
                },
            }
        yield {
            "type": "updates",
            "data": {"research_synthesize": {}},
        }
        yield {
            "type": "updates",
            "data": {"research_critic": {}},
        }
        yield {
            "type": "updates",
            "data": {"research_finalize": dict(self.result)},
        }


class ResearchExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.path = Path(self.temporary.name) / "research.sqlite"
        self.catalog = ResearchCatalog(self.path)
        self.knowledge = KnowledgeStub()
        self.research = ResearchService(self.catalog, self.knowledge)
        self.project = self.research.create_project(
            owner_user_id="user_001",
            name="执行项目",
            status="active",
            default_knowledge_base_id="kb_owned",
        )
        self.task = self.research.create_task(
            self.project.project_id,
            "user_001",
            title="分析唯一事实",
            objective="解释 RF 编号",
            task_type="synthesis",
            acceptance_criteria=["给出结论", "引用证据"],
        )

    def tearDown(self):
        self.catalog.close()
        self.temporary.cleanup()

    def execute(self, result=None, *, error=None):
        graph = FakeResearchGraph(result, error=error, catalog=self.catalog)
        service = ResearchExecutionService(self.research, graph)
        response = service.execute_task(
            project_id=self.project.project_id,
            task_id=self.task.task_id,
            user_id="user_001",
        )
        return response, graph

    def test_completed_uses_database_snapshot_binding_and_thread(self):
        response, graph = self.execute()
        self.assertEqual(response.outcome, "completed")
        self.assertEqual(response.status, "completed")
        self.assertEqual(len(graph.calls), 1)
        call = graph.calls[0]
        state = call["state"]
        run = self.research.get_run(
            self.project.project_id,
            self.task.task_id,
            response.run_id,
            "user_001",
        )
        self.assertEqual(state["task_title"], self.task.title)
        self.assertEqual(state["task_objective"], self.task.objective)
        self.assertEqual(state["acceptance_criteria"], self.task.acceptance_criteria)
        self.assertEqual(state["run_id"], run.run_id)
        self.assertEqual(call["context"].user_id, "user_001")
        self.assertEqual(
            call["config"]["configurable"],
            {"thread_id": run.thread_id, "checkpoint_ns": "researchflow"},
        )
        task = self.research.get_task(
            self.project.project_id, self.task.task_id, "user_001"
        )
        artifact = self.research.get_artifact(
            self.project.project_id,
            response.output_artifact_id,
            "user_001",
        )
        self.assertEqual(task.status, "completed")
        self.assertEqual(run.outcome, "completed")
        self.assertEqual(run.output_artifact_id, artifact.artifact_id)
        self.assertEqual(artifact.status, "draft")
        self.assertEqual(artifact.created_by, "agent")
        self.assertEqual(artifact.origin_run_id, run.run_id)

    def test_completed_sources_are_rebuilt_from_owned_ready_chunk(self):
        result = {
            "outcome": "completed",
            "final_answer": "RF-2026 是证据中的编号。",
            "sources": [
                {
                    "knowledge_base_id": "kb_owned",
                    "document_id": "forged",
                    "chunk_id": "chunk_001",
                    "source": "forged.txt",
                    "page": 999,
                    "excerpt": "伪造内容",
                }
            ],
            "unresolved_issues": [],
        }
        response, _ = self.execute(result)
        artifact = self.research.get_artifact(
            self.project.project_id,
            response.output_artifact_id,
            "user_001",
        )
        self.assertEqual(artifact.sources[0].document_id, "doc_001")
        self.assertEqual(artifact.sources[0].source, "evidence.md")
        self.assertEqual(artifact.sources[0].page, 2)
        self.assertIn("真实证据", artifact.sources[0].excerpt)

    def test_forged_cross_scope_source_fails_without_artifact(self):
        result = {
            "outcome": "completed",
            "final_answer": "伪造回答",
            "sources": [
                {"knowledge_base_id": "kb_foreign", "chunk_id": "chunk_001"}
            ],
        }
        response, graph = self.execute(result)
        self.assertEqual(len(graph.calls), 1)
        self.assertEqual(response.outcome, "failed")
        self.assertIsNone(response.output_artifact_id)
        self.assertEqual(self.catalog.list_artifacts(self.project.project_id), [])

    def test_needs_review_creates_draft_and_blocks_task(self):
        response, _ = self.execute(
            {
                "outcome": "needs_review",
                "final_answer": "安全候选答案",
                "sources": [],
                "unresolved_issues": ["样本量不足"],
            }
        )
        run = self.catalog.get_run(self.task.task_id, response.run_id)
        task = self.catalog.get_task(self.project.project_id, self.task.task_id)
        artifact = self.catalog.get_artifact(
            self.project.project_id, response.output_artifact_id
        )
        self.assertEqual((run.status, run.outcome), ("completed", "needs_review"))
        self.assertEqual(task.status, "blocked")
        self.assertEqual(artifact.status, "draft")
        self.assertIn("样本量不足", artifact.content)

    def test_blocked_and_failed_do_not_create_artifact(self):
        for outcome in ("blocked", "failed"):
            with self.subTest(outcome=outcome):
                self.task = self.research.create_task(
                    self.project.project_id,
                    "user_001",
                    title=f"{outcome}任务",
                )
                response, _ = self.execute(
                    {"outcome": outcome, "final_answer": "", "error": "证据不足"}
                )
                run = self.catalog.get_run(self.task.task_id, response.run_id)
                task = self.catalog.get_task(self.project.project_id, self.task.task_id)
                self.assertIsNone(response.output_artifact_id)
                self.assertEqual(run.outcome, outcome)
                self.assertEqual(
                    (run.status, task.status),
                    ("failed", "failed") if outcome == "failed" else ("completed", "blocked"),
                )

    def test_graph_exception_is_safely_finalized_as_failed(self):
        response, graph = self.execute(
            error=RuntimeError("API_KEY=secret; traceback internal")
        )
        run = self.catalog.get_run(self.task.task_id, response.run_id)
        task = self.catalog.get_task(self.project.project_id, self.task.task_id)
        self.assertEqual(len(graph.calls), 1)
        self.assertEqual((run.status, run.outcome), ("failed", "failed"))
        self.assertEqual(task.status, "failed")
        self.assertNotIn("secret", run.error_message)
        self.assertIsNone(run.output_artifact_id)

    def test_preconditions_and_active_run_conflict(self):
        foreign_project = self.research.create_project(
            owner_user_id="user_002", name="foreign", status="active"
        )
        foreign_task = self.research.create_task(
            foreign_project.project_id, "user_002", title="foreign"
        )
        with self.assertRaises(ResearchProjectNotFoundError):
            ResearchExecutionService(self.research, FakeResearchGraph()).execute_task(
                project_id=foreign_project.project_id,
                task_id=foreign_task.task_id,
                user_id="user_001",
            )
        with self.assertRaises(ResearchTaskNotFoundError):
            ResearchExecutionService(self.research, FakeResearchGraph()).execute_task(
                project_id=foreign_project.project_id,
                task_id=self.task.task_id,
                user_id="user_002",
            )

        self.research.start_execution(
            self.project.project_id, self.task.task_id, "user_001"
        )
        with self.assertRaises(AgentRunConflictError) as conflict:
            ResearchExecutionService(self.research, FakeResearchGraph()).execute_task(
                project_id=self.project.project_id,
                task_id=self.task.task_id,
                user_id="user_001",
            )
        self.assertIsNotNone(conflict.exception.existing_run_id)

    def test_archived_project_and_non_pending_task_cannot_start(self):
        # Archived projects reject task creation, so create then archive.
        project = self.research.create_project(
            owner_user_id="user_001", name="later archived", status="active"
        )
        task = self.research.create_task(project.project_id, "user_001", title="task")
        self.research.update_project(
            project.project_id, "user_001", status="archived"
        )
        with self.assertRaises(AgentRunConflictError):
            self.research.start_execution(project.project_id, task.task_id, "user_001")

        self.research.transition_task(
            self.project.project_id,
            self.task.task_id,
            "user_001",
            target_status="running",
        )
        with self.assertRaises(ResearchTaskConflictError):
            self.research.start_execution(
                self.project.project_id, self.task.task_id, "user_001"
            )

    def test_finish_is_idempotent_and_survives_restart(self):
        response, _ = self.execute()
        task, run, artifact = self.research.finish_execution(
            self.project.project_id,
            self.task.task_id,
            response.run_id,
            "user_001",
            outcome="completed",
            final_answer="不会产生第二份成果",
        )
        self.assertEqual(artifact.artifact_id, response.output_artifact_id)
        self.assertEqual(len(self.catalog.list_artifacts(self.project.project_id)), 1)
        self.catalog.close()
        restarted = ResearchCatalog(self.path)
        self.catalog = restarted
        self.research = ResearchService(restarted, self.knowledge)
        persisted = restarted.get_run(self.task.task_id, response.run_id)
        persisted_artifact = restarted.get_artifact_by_origin_run_id(response.run_id)
        self.assertEqual(persisted.outcome, "completed")
        self.assertEqual(persisted.output_artifact_id, persisted_artifact.artifact_id)


class ResearchExecutionApiTests(unittest.TestCase):
    def test_post_executes_and_returns_structured_result(self):
        with TemporaryDirectory() as temporary:
            catalog = ResearchCatalog(Path(temporary) / "research.sqlite")
            research = ResearchService(catalog, KnowledgeStub())
            project = research.create_project(
                owner_user_id="user_001", name="API", status="active"
            )
            task = research.create_task(project.project_id, "user_001", title="API task")
            graph = FakeResearchGraph(catalog=catalog)
            execution = ResearchExecutionService(research, graph)

            @contextmanager
            def factory():
                yield AgentService(
                    FakeResearchGraph(),
                    research_service=research,
                    research_execution_service=execution,
                )

            with TestClient(
                install_test_identity(create_app(factory)),
                raise_server_exceptions=False,
            ) as client:
                path = (
                    f"/api/v1/research/projects/{project.project_id}"
                    f"/tasks/{task.task_id}/runs"
                )
                response = client.post(path, headers={"X-User-ID": "user_001"})
                self.assertEqual(response.status_code, 201, response.text)
                payload = response.json()
                self.assertEqual(payload["outcome"], "completed")
                self.assertIsNotNone(payload["output_artifact_id"])
                self.assertNotIn("thread_id", payload)
                foreign = client.post(path, headers={"X-User-ID": "user_002"})
                self.assertEqual(foreign.status_code, 404)
            catalog.close()


class ResearchExecutionMigrationTests(unittest.TestCase):
    def test_version_four_upgrades_to_five_without_losing_run_or_artifact(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "research.sqlite"
            catalog = ResearchCatalog(path)
            service = ResearchService(catalog, KnowledgeStub())
            project = service.create_project(
                owner_user_id="user_001", name="migration", status="active"
            )
            task = service.create_task(project.project_id, "user_001", title="task")
            artifact = service.create_artifact(
                project.project_id,
                "user_001",
                task_id=task.task_id,
                title="existing artifact",
                content="existing content",
            )
            run = service.create_run(project.project_id, task.task_id, "user_001")
            catalog.close()

            connection = sqlite3.connect(path)
            connection.execute("DROP INDEX idx_research_artifacts_origin_run")
            connection.execute("DROP INDEX idx_agent_runs_output_artifact")
            connection.execute("ALTER TABLE research_artifacts DROP COLUMN origin_run_id")
            connection.execute("ALTER TABLE agent_runs DROP COLUMN outcome")
            connection.execute(
                "ALTER TABLE agent_runs RENAME COLUMN output_artifact_id TO final_artifact_id"
            )
            connection.execute("DELETE FROM schema_migrations WHERE version = 5")
            connection.commit()
            connection.close()

            upgraded = ResearchCatalog(path)
            try:
                columns = {
                    row[1]
                    for row in upgraded.connection.execute("PRAGMA table_info(agent_runs)")
                }
                artifact_columns = {
                    row[1]
                    for row in upgraded.connection.execute(
                        "PRAGMA table_info(research_artifacts)"
                    )
                }
                self.assertEqual(CURRENT_SCHEMA_VERSION, 5)
                self.assertIn("outcome", columns)
                self.assertIn("output_artifact_id", columns)
                self.assertNotIn("final_artifact_id", columns)
                self.assertIn("origin_run_id", artifact_columns)
                self.assertIsNotNone(upgraded.get_run(task.task_id, run.run_id))
                self.assertIsNotNone(
                    upgraded.get_artifact(project.project_id, artifact.artifact_id)
                )
                self.assertEqual(
                    upgraded.connection.execute(
                        "SELECT COUNT(*) FROM schema_migrations"
                    ).fetchone()[0],
                    5,
                )
            finally:
                upgraded.close()


if __name__ == "__main__":
    unittest.main()
