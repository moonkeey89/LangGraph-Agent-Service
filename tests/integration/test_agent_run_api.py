import unittest
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient

from ai_agent_learning.api.app import create_app
from ai_agent_learning.api.service import AgentService
from ai_agent_learning.knowledge.models import KnowledgeChunk
from ai_agent_learning.knowledge.service import KnowledgeNotFoundError
from ai_agent_learning.research import ResearchCatalog, ResearchService


class NoOpGraph:
    pass


class KnowledgeStub:
    def ensure_owned(self, _knowledge_base_id: str, _owner_user_id: str) -> None:
        raise KnowledgeNotFoundError

    def get_ready_chunk(self, **_kwargs) -> KnowledgeChunk:
        raise KnowledgeNotFoundError


class AgentRunApiTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.catalog = ResearchCatalog(
            Path(self.temporary.name) / "research.sqlite"
        )
        self.service = ResearchService(self.catalog, KnowledgeStub())
        self.project = self.service.create_project(
            owner_user_id="user_001",
            name="运行API项目",
            status="active",
        )
        self.task = self.service.create_task(
            self.project.project_id,
            "user_001",
            title="运行API任务",
        )
        self.second_task = self.service.create_task(
            self.project.project_id,
            "user_001",
            title="第二任务",
        )
        self.other_project = self.service.create_project(
            owner_user_id="user_001",
            name="其他项目",
            status="active",
        )
        self.other_task = self.service.create_task(
            self.other_project.project_id,
            "user_001",
            title="其他项目任务",
        )
        self.run = self.service.create_run(
            self.project.project_id,
            self.task.task_id,
            "user_001",
        )
        self.second_run = self.service.create_run(
            self.project.project_id,
            self.task.task_id,
            "user_001",
        )

        @contextmanager
        def service_factory():
            yield AgentService(NoOpGraph(), research_service=self.service)

        self.client_context = TestClient(
            create_app(service_factory),
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

    @property
    def list_path(self) -> str:
        return (
            f"/api/v1/research/projects/{self.project.project_id}"
            f"/tasks/{self.task.task_id}/runs"
        )

    def test_list_and_get_return_stable_run_response(self):
        listed = self.client.get(self.list_path, headers=self.headers())
        detail = self.client.get(
            f"{self.list_path}/{self.run.run_id}",
            headers=self.headers(),
        )
        self.assertEqual(listed.status_code, 200, listed.text)
        self.assertEqual(
            [item["attempt_number"] for item in listed.json()],
            [2, 1],
        )
        self.assertEqual(detail.status_code, 200, detail.text)
        self.assertEqual(detail.json()["run_id"], self.run.run_id)
        self.assertEqual(detail.json()["thread_id"], self.run.thread_id)
        self.assertEqual(detail.json()["status"], "pending")
        self.assertNotIn("messages", detail.json())
        self.assertNotIn("checkpoint", detail.json())

    def test_cross_user_and_mismatched_relationships_are_hidden(self):
        detail_path = f"{self.list_path}/{self.run.run_id}"
        self.assertEqual(
            self.client.get(
                self.list_path,
                headers=self.headers("user_002"),
            ).status_code,
            404,
        )
        self.assertEqual(
            self.client.get(
                detail_path,
                headers=self.headers("user_002"),
            ).status_code,
            404,
        )
        wrong_task_path = (
            f"/api/v1/research/projects/{self.project.project_id}"
            f"/tasks/{self.second_task.task_id}/runs/{self.run.run_id}"
        )
        wrong_project_path = (
            f"/api/v1/research/projects/{self.other_project.project_id}"
            f"/tasks/{self.task.task_id}/runs/{self.run.run_id}"
        )
        self.assertEqual(
            self.client.get(wrong_task_path, headers=self.headers()).status_code,
            404,
        )
        self.assertEqual(
            self.client.get(wrong_project_path, headers=self.headers()).status_code,
            404,
        )

    def test_run_routes_are_read_only(self):
        detail_path = f"{self.list_path}/{self.run.run_id}"
        for method, path in (
            ("post", self.list_path),
            ("patch", detail_path),
            ("delete", detail_path),
        ):
            if method == "delete":
                response = self.client.delete(path, headers=self.headers())
            else:
                response = getattr(self.client, method)(
                    path,
                    headers=self.headers(),
                    json={},
                )
            with self.subTest(method=method):
                self.assertEqual(response.status_code, 405, response.text)

        schema = self.client.get("/openapi.json").json()
        list_template = (
            "/api/v1/research/projects/{project_id}/tasks/{task_id}/runs"
        )
        detail_template = f"{list_template}/{{run_id}}"
        self.assertEqual(set(schema["paths"][list_template]), {"get"})
        self.assertEqual(set(schema["paths"][detail_template]), {"get"})


if __name__ == "__main__":
    unittest.main()
