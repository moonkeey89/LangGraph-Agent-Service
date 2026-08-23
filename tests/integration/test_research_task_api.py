import unittest
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient

from ai_agent_learning.api.app import create_app
from ai_agent_learning.api.service import AgentService
from ai_agent_learning.knowledge.service import KnowledgeNotFoundError
from ai_agent_learning.research import ResearchCatalog, ResearchService


class NoOpGraph:
    pass


class KnowledgeOwnershipStub:
    def ensure_owned(self, _knowledge_base_id: str, _owner_user_id: str) -> None:
        raise KnowledgeNotFoundError


class ResearchTaskApiTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.catalog = ResearchCatalog(
            Path(self.temporary.name) / "research.sqlite"
        )
        self.research_service = ResearchService(
            self.catalog,
            KnowledgeOwnershipStub(),
        )

        @contextmanager
        def service_factory():
            yield AgentService(
                NoOpGraph(),
                research_service=self.research_service,
            )

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

    def create_project(self, user_id: str = "user_001") -> dict:
        response = self.client.post(
            "/api/v1/research/projects",
            headers=self.headers(user_id),
            json={"name": f"{user_id}的项目"},
        )
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()

    def create_task(self, path_project_id: str, **overrides):
        payload = {
            "title": "整理文献",
            "objective": "总结主要方法",
            "task_type": "literature_review",
            "acceptance_criteria": ["至少三篇文献", "列出方法差异"],
        }
        payload.update(overrides)
        return self.client.post(
            f"/api/v1/research/projects/{path_project_id}/tasks",
            headers=self.headers(),
            json=payload,
        )

    def test_create_list_get_and_criteria_round_trip(self):
        project = self.create_project()
        response = self.create_task(project["project_id"])
        self.assertEqual(response.status_code, 201, response.text)
        task = response.json()
        self.assertTrue(task["task_id"].startswith("rt_"))
        self.assertEqual(task["project_id"], project["project_id"])
        self.assertEqual(task["status"], "pending")
        self.assertEqual(
            task["acceptance_criteria"],
            ["至少三篇文献", "列出方法差异"],
        )

        base = f"/api/v1/research/projects/{project['project_id']}/tasks"
        listed = self.client.get(base, headers=self.headers())
        fetched = self.client.get(
            f"{base}/{task['task_id']}",
            headers=self.headers(),
        )
        self.assertEqual(listed.status_code, 200)
        self.assertEqual(listed.json(), [task])
        self.assertEqual(fetched.json(), task)

    def test_create_cannot_set_server_controlled_fields(self):
        project = self.create_project()
        forbidden = (
            "task_id",
            "project_id",
            "status",
            "result_summary",
            "error_message",
            "started_at",
            "completed_at",
        )
        for field_name in forbidden:
            value = "running" if field_name == "status" else "forbidden"
            response = self.create_task(
                project["project_id"],
                **{field_name: value},
            )
            with self.subTest(field=field_name):
                self.assertEqual(response.status_code, 422, response.text)

    def test_cross_user_access_is_hidden_for_all_task_operations(self):
        project = self.create_project()
        task = self.create_task(project["project_id"]).json()
        other_project = self.create_project("user_002")
        base = (
            f"/api/v1/research/projects/{project['project_id']}"
            f"/tasks/{task['task_id']}"
        )
        responses = (
            self.client.get(
                f"/api/v1/research/projects/{project['project_id']}/tasks",
                headers=self.headers("user_002"),
            ),
            self.client.get(base, headers=self.headers("user_002")),
            self.client.patch(
                base,
                headers=self.headers("user_002"),
                json={"title": "越权修改"},
            ),
            self.client.post(
                f"{base}/transition",
                headers=self.headers("user_002"),
                json={"target_status": "running"},
            ),
            self.client.delete(base, headers=self.headers("user_002")),
            self.create_task(other_project["project_id"]),
        )
        for response in responses:
            self.assertEqual(response.status_code, 404, response.text)

    def test_task_cannot_be_accessed_through_another_project(self):
        project = self.create_project()
        other = self.create_project()
        task = self.create_task(project["project_id"]).json()
        path = (
            f"/api/v1/research/projects/{other['project_id']}"
            f"/tasks/{task['task_id']}"
        )
        self.assertEqual(
            self.client.get(path, headers=self.headers()).status_code,
            404,
        )

    def test_invalid_type_and_protected_patch_fields_return_422(self):
        project = self.create_project()
        invalid = self.create_task(
            project["project_id"],
            task_type="travel",
        )
        self.assertEqual(invalid.status_code, 422, invalid.text)

        task = self.create_task(project["project_id"]).json()
        path = (
            f"/api/v1/research/projects/{project['project_id']}"
            f"/tasks/{task['task_id']}"
        )
        for field_name in (
            "status",
            "task_id",
            "project_id",
            "result_summary",
            "error_message",
            "started_at",
            "completed_at",
        ):
            response = self.client.patch(
                path,
                headers=self.headers(),
                json={field_name: "forbidden"},
            )
            with self.subTest(field=field_name):
                self.assertEqual(response.status_code, 422, response.text)

    def test_partial_update_preserves_omitted_fields(self):
        project = self.create_project()
        task = self.create_task(project["project_id"]).json()
        path = (
            f"/api/v1/research/projects/{project['project_id']}"
            f"/tasks/{task['task_id']}"
        )
        response = self.client.patch(
            path,
            headers=self.headers(),
            json={"title": "更新标题"},
        )
        self.assertEqual(response.status_code, 200, response.text)
        updated = response.json()
        self.assertEqual(updated["title"], "更新标题")
        self.assertEqual(updated["objective"], task["objective"])
        self.assertEqual(updated["status"], "pending")
        self.assertEqual(
            updated["acceptance_criteria"],
            task["acceptance_criteria"],
        )
        criteria_update = self.client.patch(
            path,
            headers=self.headers(),
            json={"acceptance_criteria": ["  新验收标准  "]},
        )
        self.assertEqual(criteria_update.status_code, 200, criteria_update.text)
        self.assertEqual(
            criteria_update.json()["acceptance_criteria"],
            ["新验收标准"],
        )
        self.assertEqual(criteria_update.json()["title"], "更新标题")

    def test_transition_validation_and_conflict_status_codes(self):
        project = self.create_project()
        task = self.create_task(project["project_id"]).json()
        transition_path = (
            f"/api/v1/research/projects/{project['project_id']}"
            f"/tasks/{task['task_id']}/transition"
        )
        invalid = self.client.post(
            transition_path,
            headers=self.headers(),
            json={"target_status": "completed"},
        )
        self.assertEqual(invalid.status_code, 409, invalid.text)

        running = self.client.post(
            transition_path,
            headers=self.headers(),
            json={"target_status": "running"},
        )
        self.assertEqual(running.status_code, 200, running.text)
        missing_reason = self.client.post(
            transition_path,
            headers=self.headers(),
            json={"target_status": "failed"},
        )
        self.assertEqual(missing_reason.status_code, 422, missing_reason.text)
        failed = self.client.post(
            transition_path,
            headers=self.headers(),
            json={"target_status": "failed", "reason": "资料损坏"},
        )
        self.assertEqual(failed.status_code, 200, failed.text)
        self.assertEqual(failed.json()["error_message"], "资料损坏")

    def test_project_with_task_cannot_be_deleted_and_task_delete_is_restricted(self):
        project = self.create_project()
        task = self.create_task(project["project_id"]).json()
        project_path = f"/api/v1/research/projects/{project['project_id']}"
        self.assertEqual(
            self.client.delete(project_path, headers=self.headers()).status_code,
            409,
        )

        task_path = f"{project_path}/tasks/{task['task_id']}"
        running = self.client.post(
            f"{task_path}/transition",
            headers=self.headers(),
            json={"target_status": "running"},
        )
        self.assertEqual(running.status_code, 200)
        self.assertEqual(
            self.client.delete(task_path, headers=self.headers()).status_code,
            409,
        )

    def test_archived_project_allows_task_reads_but_no_mutations(self):
        project = self.create_project()
        task = self.create_task(project["project_id"]).json()
        project_path = f"/api/v1/research/projects/{project['project_id']}"
        archived = self.client.patch(
            project_path,
            headers=self.headers(),
            json={"status": "archived"},
        )
        self.assertEqual(archived.status_code, 200)

        task_path = f"{project_path}/tasks/{task['task_id']}"
        self.assertEqual(
            self.client.get(task_path, headers=self.headers()).status_code,
            200,
        )
        mutations = (
            self.create_task(project["project_id"]),
            self.client.patch(
                task_path,
                headers=self.headers(),
                json={"title": "不可修改"},
            ),
            self.client.post(
                f"{task_path}/transition",
                headers=self.headers(),
                json={"target_status": "running"},
            ),
            self.client.delete(task_path, headers=self.headers()),
        )
        for response in mutations:
            self.assertEqual(response.status_code, 409, response.text)


if __name__ == "__main__":
    unittest.main()
