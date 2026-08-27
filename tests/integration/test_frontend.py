import unittest
from contextlib import contextmanager

from fastapi.testclient import TestClient

from ai_agent_learning.api.app import create_app
from ai_agent_learning.api.service import AgentService
from tests.helpers import install_test_identity


class NeverCalledGraph:
    def get_state(self, _config):
        raise AssertionError("Frontend asset tests must not call the graph")


class FrontendIntegrationTests(unittest.TestCase):
    def setUp(self):
        @contextmanager
        def service_factory():
            yield AgentService(NeverCalledGraph())

        self.client_context = TestClient(
            install_test_identity(create_app(service_factory))
        )
        self.client = self.client_context.__enter__()

    def tearDown(self):
        self.client_context.__exit__(None, None, None)

    def test_chat_page_and_assets_are_available(self):
        page = self.client.get("/")
        script = self.client.get("/assets/app.js")
        parser = self.client.get("/assets/sse-parser.js")
        api_module = self.client.get("/assets/api.js")
        auth_state = self.client.get("/assets/auth-state.js")
        research_module = self.client.get("/assets/researchflow.js")
        research_state = self.client.get("/assets/researchflow-state.js")
        styles = self.client.get("/assets/styles.css")

        self.assertEqual(page.status_code, 200)
        self.assertIn("text/html", page.headers["content-type"])
        self.assertIn("AI Agent Learning", page.text)
        self.assertIn("ResearchFlow", page.text)
        self.assertIn('id="initializing-view"', page.text)
        self.assertIn('id="auth-view"', page.text)
        self.assertIn('id="workspace-root"', page.text)
        self.assertIn('id="login-form"', page.text)
        self.assertIn('id="register-form"', page.text)
        self.assertIn('id="logout-button"', page.text)
        self.assertIn('/assets/app.js', page.text)
        self.assertIn('/assets/styles.css', page.text)

        self.assertEqual(script.status_code, 200)
        self.assertIn("javascript", script.headers["content-type"])
        self.assertIn('/api/v1/agent/stream', script.text)
        self.assertIn('/api/v1/agent/resume', script.text)
        self.assertIn("AbortController", script.text)
        self.assertIn("response.body.getReader()", script.text)
        self.assertIn("renderSources", script.text)
        self.assertIn("message-sources", script.text)
        self.assertIn("knowledge-view", page.text)
        self.assertIn("knowledge-selector", page.text)
        self.assertIn("/api/v1/knowledge-bases", script.text)
        self.assertIn("FormData", script.text)
        self.assertIn("dragover", script.text)
        self.assertIn("initResearchFlow", script.text)
        self.assertIn('/api/v1/auth/me', script.text)
        self.assertIn('/api/v1/auth/register', script.text)
        self.assertIn('/api/v1/auth/login', script.text)
        self.assertIn('/api/v1/auth/logout', script.text)
        self.assertIn("clearBusinessState", script.text)
        self.assertIn("restoreSession", script.text)

        self.assertEqual(api_module.status_code, 200)
        self.assertIn("ApiError", api_module.text)
        self.assertIn('credentials: "same-origin"', api_module.text)
        self.assertIn("X-CSRF-Token", api_module.text)
        self.assertIn("authenticatedHeaders", api_module.text)

        self.assertEqual(auth_state.status_code, 200)
        self.assertIn("validRegistration", auth_state.text)
        self.assertIn("runAuthStateSelfTests", auth_state.text)

        self.assertEqual(research_module.status_code, 200)
        self.assertIn("/runs/stream", research_module.text)
        self.assertIn("run_started", research_module.text)
        self.assertIn("run_completed", research_module.text)
        self.assertIn("run_blocked", research_module.text)
        self.assertIn("run_needs_review", research_module.text)
        self.assertIn("run_failed", research_module.text)
        self.assertIn("createSseParser", research_module.text)
        self.assertIn("loadProjectWorkspace", research_module.text)
        self.assertIn("定稿后不能直接修改或删除", research_module.text)

        self.assertEqual(research_state.status_code, 200)
        self.assertIn("isTaskStartable", research_state.text)
        self.assertIn("appendStreamToken", research_state.text)
        self.assertIn("single-token-buffer", research_state.text)

        self.assertEqual(parser.status_code, 200)
        self.assertIn("createSseParser", parser.text)
        self.assertIn("runSseParserSelfTests", parser.text)
        self.assertIn("utf8-cross-byte-boundary", parser.text)
        self.assertIn("invalid-json-recovery", parser.text)

        self.assertEqual(styles.status_code, 200)
        self.assertIn("text/css", styles.headers["content-type"])

    def test_frontend_does_not_shadow_docs_or_api_routes(self):
        docs = self.client.get("/docs")
        redoc = self.client.get("/redoc")
        health = self.client.get("/health")
        invalid_invoke = self.client.post("/api/v1/agent/invoke", json={})
        openapi = self.client.get("/openapi.json").json()

        self.assertEqual(docs.status_code, 200)
        self.assertIn("swagger-ui", docs.text)
        self.assertEqual(redoc.status_code, 200)
        self.assertEqual(health.json(), {"status": "ok"})
        self.assertEqual(invalid_invoke.status_code, 422)
        self.assertIn("/api/v1/agent/invoke", openapi["paths"])
        self.assertIn("/api/v1/agent/resume", openapi["paths"])
        self.assertIn("/api/v1/agent/stream", openapi["paths"])
        self.assertIn("/api/v1/knowledge-bases", openapi["paths"])
        self.assertIn("/api/v1/research/projects", openapi["paths"])
        self.assertIn(
            "/api/v1/research/projects/{project_id}/tasks/{task_id}/runs/stream",
            openapi["paths"],
        )

    def test_frontend_uses_safe_text_rendering_and_no_secret(self):
        script = self.client.get("/assets/app.js").text
        api_script = self.client.get("/assets/api.js").text
        research_script = self.client.get("/assets/researchflow.js").text
        page = self.client.get("/").text

        self.assertNotIn("innerHTML", script)
        self.assertNotIn("innerHTML", research_script)
        self.assertIn("textContent", script)
        self.assertIn("textContent", research_script)
        self.assertNotIn("DEEPSEEK_API_KEY", script)
        self.assertNotIn("DEEPSEEK_API_KEY", page)
        self.assertNotIn("DEEPSEEK_API_KEY", research_script)
        self.assertNotIn("sk-", script)
        self.assertNotIn("X-User-ID", page)
        self.assertNotIn("X-User-ID", script)
        self.assertNotIn("X-User-ID", api_script)
        self.assertNotIn("X-User-ID", research_script)
        self.assertNotIn('id="user-id"', page)
        self.assertNotIn("STORAGE_USER_ID", script)
        self.assertNotIn("sessionStorage", script)
        self.assertNotIn("session_token", script.lower())

    def test_auth_gate_prevents_eager_business_loading(self):
        page = self.client.get("/").text
        script = self.client.get("/assets/app.js").text
        research_script = self.client.get("/assets/researchflow.js").text

        self.assertIn('id="workspace-root" class="app-shell" aria-labelledby="page-title" hidden', page)
        self.assertIn('id="auth-view" class="auth-shell" hidden', page)
        self.assertIn('apiFetch("/api/v1/auth/me"', script)
        self.assertIn("await enterWorkspace(user)", script)
        self.assertIn("enabled: false", research_script)
        self.assertIn("async function start()", research_script)
        self.assertIn("await loadProjects()", research_script)

    def test_all_business_requests_use_the_shared_authenticated_client(self):
        script = self.client.get("/assets/app.js").text
        research_script = self.client.get("/assets/researchflow.js").text
        api_script = self.client.get("/assets/api.js").text

        self.assertNotIn("fetch(\"/api/v1", script)
        self.assertNotIn("fetch(\"/api/v1", research_script)
        self.assertEqual(api_script.count("fetch(path"), 1)
        self.assertIn("apiFetch(\"/api/v1/agent/stream\"", script)
        self.assertIn("apiFetch(\"/api/v1/agent/resume\"", script)
        self.assertIn("/runs/stream", research_script)

    def test_logout_and_session_failure_clear_user_scoped_browser_state(self):
        script = self.client.get("/assets/app.js").text
        research_script = self.client.get("/assets/researchflow.js").text

        self.assertIn('apiFetch("/api/v1/auth/logout"', script)
        self.assertIn("state.controller?.abort()", script)
        self.assertIn("state.knowledgeBases = []", script)
        self.assertIn("state.knowledgeDocuments = []", script)
        self.assertIn("localStorage.removeItem(STORAGE_THREAD_ID)", script)
        self.assertIn("localStorage.removeItem(STORAGE_KNOWLEDGE_BASE_ID)", script)
        self.assertIn("localStorage.removeItem(STORAGE_PROJECT_ID)", script)
        self.assertIn("researchFlow?.reset()", script)
        self.assertIn("state.streamController?.abort()", research_script)
        self.assertIn("elements.runTimeline.replaceChildren()", research_script)

    def test_research_workbench_structure_is_available(self):
        page = self.client.get("/").text

        for element_id in (
            "nav-overview",
            "nav-knowledge",
            "nav-tasks",
            "nav-artifacts",
            "nav-chat",
            "project-selector",
            "project-list",
            "task-list",
            "run-timeline",
            "artifact-list",
            "artifact-sources",
        ):
            self.assertIn(f'id="{element_id}"', page)


if __name__ == "__main__":
    unittest.main()
