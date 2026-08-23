import unittest
from contextlib import contextmanager

from fastapi.testclient import TestClient

from ai_agent_learning.api.app import create_app
from ai_agent_learning.api.service import AgentService


class NeverCalledGraph:
    def get_state(self, _config):
        raise AssertionError("Frontend asset tests must not call the graph")


class FrontendIntegrationTests(unittest.TestCase):
    def setUp(self):
        @contextmanager
        def service_factory():
            yield AgentService(NeverCalledGraph())

        self.client_context = TestClient(create_app(service_factory))
        self.client = self.client_context.__enter__()

    def tearDown(self):
        self.client_context.__exit__(None, None, None)

    def test_chat_page_and_assets_are_available(self):
        page = self.client.get("/")
        script = self.client.get("/assets/app.js")
        parser = self.client.get("/assets/sse-parser.js")
        styles = self.client.get("/assets/styles.css")

        self.assertEqual(page.status_code, 200)
        self.assertIn("text/html", page.headers["content-type"])
        self.assertIn("AI Agent Learning", page.text)
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

    def test_frontend_uses_safe_text_rendering_and_no_secret(self):
        script = self.client.get("/assets/app.js").text
        page = self.client.get("/").text

        self.assertNotIn("innerHTML", script)
        self.assertIn("textContent", script)
        self.assertNotIn("DEEPSEEK_API_KEY", script)
        self.assertNotIn("DEEPSEEK_API_KEY", page)
        self.assertNotIn("sk-", script)
        self.assertNotIn("innerHTML", script)


if __name__ == "__main__":
    unittest.main()
