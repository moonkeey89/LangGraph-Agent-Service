import unittest
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient

from ai_agent_learning.api.app import create_app
from ai_agent_learning.api.service import AgentService
from tests.helpers import install_test_identity
from ai_agent_learning.knowledge.models import KnowledgeChunk
from ai_agent_learning.knowledge.service import KnowledgeNotFoundError
from ai_agent_learning.research import ResearchCatalog, ResearchService


class NoOpGraph:
    pass


class EvidenceResolverStub:
    def __init__(self):
        self.owners = {"kb-user-001": "user_001", "kb-user-002": "user_002"}
        self.chunks = {
            ("kb-user-001", "chunk-owned"): KnowledgeChunk(
                content="可信实验结果为91%。",
                knowledge_base_id="kb-user-001",
                document_id="doc-owned",
                source="evidence.md",
                page=3,
                chunk_id="chunk-owned",
            ),
            ("kb-user-002", "chunk-private"): KnowledgeChunk(
                content="其他用户的证据。",
                knowledge_base_id="kb-user-002",
                document_id="doc-private",
                source="private.md",
                page=None,
                chunk_id="chunk-private",
            ),
        }

    def ensure_owned(self, knowledge_base_id: str, owner_user_id: str) -> None:
        if self.owners.get(knowledge_base_id) != owner_user_id:
            raise KnowledgeNotFoundError

    def get_ready_chunk(
        self,
        *,
        knowledge_base_id: str,
        owner_user_id: str,
        chunk_id: str,
    ) -> KnowledgeChunk:
        self.ensure_owned(knowledge_base_id, owner_user_id)
        chunk = self.chunks.get((knowledge_base_id, chunk_id))
        if chunk is None:
            raise KnowledgeNotFoundError
        return chunk


class ResearchArtifactApiTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.catalog = ResearchCatalog(
            Path(self.temporary.name) / "research.sqlite"
        )
        self.evidence = EvidenceResolverStub()
        self.research_service = ResearchService(self.catalog, self.evidence)

        @contextmanager
        def service_factory():
            yield AgentService(
                NoOpGraph(),
                research_service=self.research_service,
            )

        self.client_context = TestClient(
            install_test_identity(create_app(service_factory)),
            raise_server_exceptions=False,
        )
        self.client = self.client_context.__enter__()

    def tearDown(self):
        self.client_context.__exit__(None, None, None)
        self.catalog.close()
        self.temporary.cleanup()

    @staticmethod
    def headers(user_id: str = "user_001") -> dict[str, str]:
        return {"X-User-ID": user_id}

    def create_project(
        self,
        user_id: str = "user_001",
        *,
        knowledge_base_id: str | None = "kb-user-001",
    ) -> dict:
        response = self.client.post(
            "/api/v1/research/projects",
            headers=self.headers(user_id),
            json={
                "name": f"{user_id}项目",
                "status": "active",
                "default_knowledge_base_id": knowledge_base_id,
            },
        )
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()

    def create_task(self, project_id: str, user_id: str = "user_001") -> dict:
        response = self.client.post(
            f"/api/v1/research/projects/{project_id}/tasks",
            headers=self.headers(user_id),
            json={"title": "分析任务", "task_type": "analysis"},
        )
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()

    def create_artifact(
        self,
        path_project_id: str,
        user_id: str = "user_001",
        **overrides,
    ):
        payload = {
            "title": "阶段分析",
            "content": "这是分析草稿。",
            "artifact_type": "analysis",
        }
        payload.update(overrides)
        return self.client.post(
            f"/api/v1/research/projects/{path_project_id}/artifacts",
            headers=self.headers(user_id),
            json=payload,
        )

    def test_create_list_get_filter_and_source_response(self):
        project = self.create_project()
        task = self.create_task(project["project_id"])
        response = self.create_artifact(
            project["project_id"],
            task_id=task["task_id"],
            source_chunk_ids=["chunk-owned"],
        )
        self.assertEqual(response.status_code, 201, response.text)
        artifact = response.json()
        self.assertTrue(artifact["artifact_id"].startswith("ra_"))
        self.assertEqual(artifact["status"], "draft")
        self.assertEqual(artifact["created_by"], "user")
        self.assertEqual(artifact["sources"][0]["source"], "evidence.md")
        self.assertEqual(artifact["sources"][0]["page"], 3)
        self.assertEqual(artifact["sources"][0]["excerpt"], "可信实验结果为91%。")

        base = f"/api/v1/research/projects/{project['project_id']}/artifacts"
        listed = self.client.get(
            base,
            headers=self.headers(),
            params={
                "task_id": task["task_id"],
                "artifact_type": "analysis",
                "status": "draft",
            },
        )
        fetched = self.client.get(
            f"{base}/{artifact['artifact_id']}",
            headers=self.headers(),
        )
        self.assertEqual(listed.status_code, 200, listed.text)
        self.assertEqual(listed.json(), [artifact])
        self.assertEqual(fetched.json(), artifact)

    def test_create_rejects_server_controlled_fields_and_cross_project_task(self):
        project = self.create_project()
        other_project = self.create_project(knowledge_base_id=None)
        other_task = self.create_task(other_project["project_id"])
        wrong_task = self.create_artifact(
            project["project_id"],
            task_id=other_task["task_id"],
        )
        self.assertEqual(wrong_task.status_code, 404, wrong_task.text)

        forbidden = (
            "artifact_id",
            "project_id",
            "status",
            "created_by",
            "created_at",
            "updated_at",
            "finalized_at",
        )
        for field_name in forbidden:
            response = self.create_artifact(
                project["project_id"],
                **{field_name: "agent" if field_name == "created_by" else "forbidden"},
            )
            with self.subTest(field=field_name):
                self.assertEqual(response.status_code, 422, response.text)

    def test_missing_and_other_user_source_chunks_are_hidden(self):
        project = self.create_project()
        for chunk_id in ("missing", "chunk-private"):
            response = self.create_artifact(
                project["project_id"],
                source_chunk_ids=[chunk_id],
            )
            with self.subTest(chunk_id=chunk_id):
                self.assertEqual(response.status_code, 404, response.text)

    def test_cross_user_and_wrong_project_access_are_hidden(self):
        project = self.create_project()
        artifact = self.create_artifact(project["project_id"]).json()
        other_project = self.create_project(knowledge_base_id=None)
        path = (
            f"/api/v1/research/projects/{project['project_id']}"
            f"/artifacts/{artifact['artifact_id']}"
        )
        responses = (
            self.client.get(path, headers=self.headers("user_002")),
            self.client.patch(
                path,
                headers=self.headers("user_002"),
                json={"title": "越权"},
            ),
            self.client.post(
                f"{path}/finalize",
                headers=self.headers("user_002"),
            ),
            self.client.delete(path, headers=self.headers("user_002")),
            self.client.get(
                f"/api/v1/research/projects/{other_project['project_id']}"
                f"/artifacts/{artifact['artifact_id']}",
                headers=self.headers(),
            ),
        )
        for response in responses:
            self.assertEqual(response.status_code, 404, response.text)

    def test_draft_update_finalize_and_final_protection(self):
        project = self.create_project()
        artifact = self.create_artifact(project["project_id"]).json()
        path = (
            f"/api/v1/research/projects/{project['project_id']}"
            f"/artifacts/{artifact['artifact_id']}"
        )
        updated = self.client.patch(
            path,
            headers=self.headers(),
            json={
                "title": "正式报告",
                "artifact_type": "report",
                "content": "修订后的报告正文。",
                "source_chunk_ids": ["chunk-owned"],
            },
        )
        self.assertEqual(updated.status_code, 200, updated.text)
        self.assertEqual(updated.json()["status"], "draft")
        finalized = self.client.post(f"{path}/finalize", headers=self.headers())
        self.assertEqual(finalized.status_code, 200, finalized.text)
        self.assertEqual(finalized.json()["status"], "final")
        self.assertIsNotNone(finalized.json()["finalized_at"])
        self.assertEqual(
            self.client.post(f"{path}/finalize", headers=self.headers()).status_code,
            409,
        )
        self.assertEqual(
            self.client.patch(
                path,
                headers=self.headers(),
                json={"title": "不可修改"},
            ).status_code,
            409,
        )
        self.assertEqual(
            self.client.delete(path, headers=self.headers()).status_code,
            409,
        )
        self.assertEqual(
            self.client.patch(
                path,
                headers=self.headers(),
                json={"status": "draft"},
            ).status_code,
            422,
        )

    def test_draft_delete_and_parent_delete_restrictions(self):
        project = self.create_project()
        task = self.create_task(project["project_id"])
        artifact = self.create_artifact(
            project["project_id"],
            task_id=task["task_id"],
        ).json()
        project_path = f"/api/v1/research/projects/{project['project_id']}"
        self.assertEqual(
            self.client.delete(project_path, headers=self.headers()).status_code,
            409,
        )
        self.assertEqual(
            self.client.delete(
                f"{project_path}/tasks/{task['task_id']}",
                headers=self.headers(),
            ).status_code,
            409,
        )
        self.assertEqual(
            self.client.delete(
                f"{project_path}/artifacts/{artifact['artifact_id']}",
                headers=self.headers(),
            ).status_code,
            204,
        )
        self.assertEqual(
            self.client.get(
                f"{project_path}/tasks/{task['task_id']}",
                headers=self.headers(),
            ).status_code,
            200,
        )

    def test_archived_project_allows_reads_only(self):
        project = self.create_project()
        artifact = self.create_artifact(project["project_id"]).json()
        project_path = f"/api/v1/research/projects/{project['project_id']}"
        self.assertEqual(
            self.client.patch(
                project_path,
                headers=self.headers(),
                json={"status": "archived"},
            ).status_code,
            200,
        )
        artifact_path = f"{project_path}/artifacts/{artifact['artifact_id']}"
        self.assertEqual(
            self.client.get(artifact_path, headers=self.headers()).status_code,
            200,
        )
        mutations = (
            self.create_artifact(project["project_id"]),
            self.client.patch(
                artifact_path,
                headers=self.headers(),
                json={"title": "不可修改"},
            ),
            self.client.post(
                f"{artifact_path}/finalize",
                headers=self.headers(),
            ),
            self.client.delete(artifact_path, headers=self.headers()),
        )
        for response in mutations:
            self.assertEqual(response.status_code, 409, response.text)


if __name__ == "__main__":
    unittest.main()
