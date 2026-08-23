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
    def __init__(self):
        self.owners = {
            "kb_user_001": "user_001",
            "kb_user_002": "user_002",
        }

    def ensure_owned(self, knowledge_base_id: str, owner_user_id: str) -> None:
        if self.owners.get(knowledge_base_id) != owner_user_id:
            raise KnowledgeNotFoundError


class ResearchProjectApiTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.catalog = ResearchCatalog(
            Path(self.temporary.name) / "research.sqlite"
        )
        self.knowledge = KnowledgeOwnershipStub()
        self.research_service = ResearchService(
            self.catalog,
            self.knowledge,
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

    def create_project(self, **overrides):
        payload = {
            "name": "科研项目",
            "description": "初始说明",
            "research_question": "初始问题",
        }
        payload.update(overrides)
        return self.client.post(
            "/api/v1/research/projects",
            headers=self.headers(),
            json=payload,
        )

    def test_create_list_and_get_project(self):
        response = self.create_project()
        self.assertEqual(response.status_code, 201, response.text)
        project = response.json()
        self.assertTrue(project["project_id"].startswith("rp_"))
        self.assertEqual(project["owner_user_id"], "user_001")

        listed = self.client.get(
            "/api/v1/research/projects",
            headers=self.headers(),
        )
        fetched = self.client.get(
            f"/api/v1/research/projects/{project['project_id']}",
            headers=self.headers(),
        )
        self.assertEqual(listed.status_code, 200)
        self.assertEqual(len(listed.json()), 1)
        self.assertEqual(fetched.json(), project)

    def test_client_cannot_set_project_or_owner_identity(self):
        for forbidden_field in ("project_id", "owner_user_id"):
            response = self.create_project(**{forbidden_field: "attacker"})
            with self.subTest(field=forbidden_field):
                self.assertEqual(response.status_code, 422, response.text)

        project = self.create_project().json()
        for forbidden_field in ("project_id", "owner_user_id"):
            response = self.client.patch(
                f"/api/v1/research/projects/{project['project_id']}",
                headers=self.headers(),
                json={forbidden_field: "attacker"},
            )
            with self.subTest(field=forbidden_field):
                self.assertEqual(response.status_code, 422, response.text)

    def test_foreign_project_get_patch_delete_all_return_404(self):
        project = self.create_project().json()
        path = f"/api/v1/research/projects/{project['project_id']}"
        responses = (
            self.client.get(path, headers=self.headers("user_002")),
            self.client.patch(
                path,
                headers=self.headers("user_002"),
                json={"name": "越权"},
            ),
            self.client.delete(path, headers=self.headers("user_002")),
        )
        for response in responses:
            self.assertEqual(response.status_code, 404, response.text)

        self.assertEqual(
            self.client.get(path, headers=self.headers()).json()["name"],
            "科研项目",
        )

    def test_patch_updates_fields_and_preserves_omitted_values(self):
        project = self.create_project().json()
        path = f"/api/v1/research/projects/{project['project_id']}"
        response = self.client.patch(
            path,
            headers=self.headers(),
            json={
                "name": "更新项目",
                "description": "更新说明",
                "research_question": "更新问题",
                "status": "active",
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        updated = response.json()
        self.assertEqual(updated["status"], "active")

        partial = self.client.patch(
            path,
            headers=self.headers(),
            json={"name": "仅更新名称"},
        ).json()
        self.assertEqual(partial["name"], "仅更新名称")
        self.assertEqual(partial["description"], "更新说明")
        self.assertEqual(partial["research_question"], "更新问题")
        self.assertEqual(partial["status"], "active")

    def test_invalid_status_blank_name_empty_patch_and_invalid_null_are_422(self):
        blank = self.create_project(name="   ")
        invalid_status = self.create_project(status="unknown")
        self.assertEqual(blank.status_code, 422, blank.text)
        self.assertEqual(invalid_status.status_code, 422, invalid_status.text)

        project = self.create_project().json()
        path = f"/api/v1/research/projects/{project['project_id']}"
        payloads = ({}, {"name": None}, {"description": None})
        for payload in payloads:
            response = self.client.patch(
                path,
                headers=self.headers(),
                json=payload,
            )
            with self.subTest(payload=payload):
                self.assertEqual(response.status_code, 422, response.text)

    def test_owned_knowledge_base_can_be_bound_and_explicitly_unbound(self):
        response = self.create_project(
            default_knowledge_base_id="kb_user_001"
        )
        self.assertEqual(response.status_code, 201, response.text)
        project = response.json()
        self.assertEqual(
            project["default_knowledge_base_id"],
            "kb_user_001",
        )

        updated = self.client.patch(
            f"/api/v1/research/projects/{project['project_id']}",
            headers=self.headers(),
            json={"default_knowledge_base_id": None},
        )
        self.assertEqual(updated.status_code, 200, updated.text)
        self.assertIsNone(updated.json()["default_knowledge_base_id"])

    def test_foreign_knowledge_base_binding_is_hidden(self):
        response = self.create_project(
            default_knowledge_base_id="kb_user_002"
        )
        self.assertEqual(response.status_code, 404, response.text)

        project = self.create_project().json()
        updated = self.client.patch(
            f"/api/v1/research/projects/{project['project_id']}",
            headers=self.headers(),
            json={"default_knowledge_base_id": "kb_user_002"},
        )
        self.assertEqual(updated.status_code, 404, updated.text)

    def test_delete_project_returns_204_and_does_not_delete_knowledge_base(self):
        project = self.create_project(
            default_knowledge_base_id="kb_user_001"
        ).json()
        path = f"/api/v1/research/projects/{project['project_id']}"
        response = self.client.delete(path, headers=self.headers())

        self.assertEqual(response.status_code, 204, response.text)
        self.assertEqual(
            self.client.get(path, headers=self.headers()).status_code,
            404,
        )
        self.assertEqual(self.knowledge.owners["kb_user_001"], "user_001")

    def test_openapi_documents_research_business_responses(self):
        document = self.client.get("/openapi.json").json()
        operation = document["paths"]["/api/v1/research/projects/{project_id}"][
            "patch"
        ]
        self.assertIn("404", operation["responses"])
        self.assertIn("409", operation["responses"])
        self.assertIn("422", operation["responses"])


if __name__ == "__main__":
    unittest.main()
