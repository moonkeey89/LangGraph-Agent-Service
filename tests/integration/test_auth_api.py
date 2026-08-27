import unittest
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from ai_agent_learning.api.app import create_app
from ai_agent_learning.api.service import AgentService
from ai_agent_learning.auth import (
    AuthCatalog,
    AuthCookieConfig,
    AuthenticationRequiredError,
    AuthService,
)


class AuthGraph:
    def __init__(self):
        self.owners = {}
        self.calls = []

    def get_state(self, config):
        thread_id = config["configurable"]["thread_id"]
        owner = self.owners.get(thread_id)
        return SimpleNamespace(values={"session_user_id": owner} if owner else {})

    def invoke(self, state, *, config, context):
        thread_id = config["configurable"]["thread_id"]
        self.owners[thread_id] = context.user_id
        self.calls.append({"thread_id": thread_id, "user_id": context.user_id})
        return {
            "messages": [AIMessage(content="认证调用成功")],
            "session_user_id": context.user_id,
        }


class AuthApiTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.catalog = AuthCatalog(Path(self.temporary.name) / "auth.sqlite")
        self.config = AuthCookieConfig(
            session_cookie_name="test_session",
            csrf_cookie_name="test_csrf",
            csrf_header_name="X-CSRF-Token",
            secure=False,
            same_site="lax",
            domain=None,
            session_ttl_minutes=60,
        )
        self.auth = AuthService(self.catalog, self.config)
        self.graph = AuthGraph()

        @contextmanager
        def factory():
            yield AgentService(self.graph, auth_service=self.auth)

        self.client_context = TestClient(
            create_app(factory),
            raise_server_exceptions=False,
        )
        self.client = self.client_context.__enter__()

    def tearDown(self):
        self.client_context.__exit__(None, None, None)
        self.catalog.close()
        self.temporary.cleanup()

    def register(self, username="alice", email="alice@example.com", password="correct-password"):
        return self.client.post(
            "/api/v1/auth/register",
            json={"username": username, "email": email, "password": password},
        )

    def login(self, login="alice", password="correct-password"):
        return self.client.post(
            "/api/v1/auth/login",
            json={"login": login, "password": password},
        )

    def csrf_headers(self, **extra):
        return {"X-CSRF-Token": self.client.cookies.get("test_csrf"), **extra}

    def invoke(self, thread_id="auth-thread", headers=None):
        return self.client.post(
            "/api/v1/agent/invoke",
            headers=headers or {},
            json={"message": "你好", "thread_id": thread_id},
        )

    def test_register_login_cookie_and_me(self):
        self.assertEqual(self.client.get("/api/v1/auth/me").status_code, 401)
        registered = self.register()
        self.assertEqual(registered.status_code, 201, registered.text)
        self.assertNotIn("password", registered.text.casefold())
        self.assertIsNone(self.client.cookies.get("test_session"))
        logged_in = self.login()
        self.assertEqual(logged_in.status_code, 200, logged_in.text)
        cookies = logged_in.headers.get_list("set-cookie")
        session_cookie = next(item for item in cookies if item.startswith("test_session="))
        csrf_cookie = next(item for item in cookies if item.startswith("test_csrf="))
        self.assertIn("HttpOnly", session_cookie)
        self.assertIn("SameSite=lax", session_cookie)
        self.assertNotIn("HttpOnly", csrf_cookie)
        self.assertNotIn("session_token", logged_in.text)
        self.assertEqual(
            logged_in.headers["X-ResearchFlow-CSRF-Cookie"],
            "test_csrf",
        )
        self.assertEqual(
            logged_in.headers["X-ResearchFlow-CSRF-Header"],
            "X-CSRF-Token",
        )
        me = self.client.get("/api/v1/auth/me")
        self.assertEqual(me.status_code, 200)
        self.assertEqual(me.json()["username"], "alice")
        self.assertEqual(me.headers["X-ResearchFlow-CSRF-Cookie"], "test_csrf")

    def test_duplicate_and_bad_login_have_safe_responses(self):
        self.assertEqual(self.register().status_code, 201)
        duplicate_username = self.register(
            email="another@example.com"
        )
        duplicate_email = self.register(
            username="another"
        )
        self.assertEqual(duplicate_username.status_code, 409)
        self.assertEqual(duplicate_email.status_code, 409)
        wrong_password = self.login(password="wrong-password")
        missing_user = self.login(login="missing", password="wrong-password")
        self.assertEqual(wrong_password.status_code, 401)
        self.assertEqual(missing_user.status_code, 401)
        self.assertEqual(wrong_password.json(), missing_user.json())

    def test_business_api_requires_cookie_and_ignores_forged_header(self):
        unauthenticated = self.invoke(headers={"X-User-ID": "forged"})
        self.assertEqual(unauthenticated.status_code, 401)
        self.register()
        self.login()
        response = self.invoke(
            headers=self.csrf_headers(**{"X-User-ID": "forged"})
        )
        self.assertEqual(response.status_code, 200, response.text)
        authenticated_user = self.client.get("/api/v1/auth/me").json()["user_id"]
        self.assertEqual(self.graph.calls[-1]["user_id"], authenticated_user)
        self.assertNotEqual(self.graph.calls[-1]["user_id"], "forged")

        forged_body = self.client.post(
            "/api/v1/agent/invoke",
            headers=self.csrf_headers(),
            json={
                "message": "你好",
                "thread_id": "forged-body",
                "user_id": "forged",
            },
        )
        self.assertEqual(forged_body.status_code, 422)

    def test_csrf_is_required_for_cookie_authenticated_mutations(self):
        self.register()
        self.login()
        self.assertEqual(self.invoke().status_code, 403)
        self.assertEqual(
            self.invoke(headers={"X-CSRF-Token": "wrong"}).status_code,
            403,
        )
        self.assertEqual(self.invoke(headers=self.csrf_headers()).status_code, 200)

    def test_expired_inactive_and_revoked_sessions_fail(self):
        self.register()
        self.login()
        token = self.client.cookies.get("test_session")
        session = self.auth.authenticate(token).session
        with self.catalog.connection:
            self.catalog.connection.execute(
                "UPDATE sessions SET expires_at = ? WHERE session_id = ?",
                ("2000-01-01T00:00:00+00:00", session.session_id),
            )
        self.assertEqual(self.client.get("/api/v1/auth/me").status_code, 401)

        self.client.cookies.clear()
        self.login()
        user_id = self.client.get("/api/v1/auth/me").json()["user_id"]
        with self.catalog.connection:
            self.catalog.connection.execute(
                "UPDATE users SET is_active = 0 WHERE user_id = ?",
                (user_id,),
            )
        self.assertEqual(self.client.get("/api/v1/auth/me").status_code, 401)

    def test_logout_revokes_server_session_and_clears_cookies(self):
        self.register()
        self.login()
        session_token = self.client.cookies.get("test_session")
        csrf = self.client.cookies.get("test_csrf")
        response = self.client.post(
            "/api/v1/auth/logout",
            headers={"X-CSRF-Token": csrf},
        )
        self.assertEqual(response.status_code, 204, response.text)
        self.assertIsNone(self.client.cookies.get("test_session"))
        self.assertIsNone(self.client.cookies.get("test_csrf"))
        with self.assertRaises(AuthenticationRequiredError):
            self.auth.authenticate(session_token)

    def test_logout_requires_csrf(self):
        self.register()
        self.login()
        missing = self.client.post("/api/v1/auth/logout")
        wrong = self.client.post(
            "/api/v1/auth/logout",
            headers={"X-CSRF-Token": "wrong"},
        )
        self.assertEqual(missing.status_code, 403)
        self.assertEqual(wrong.status_code, 403)
        self.assertEqual(self.client.get("/api/v1/auth/me").status_code, 200)

    def test_cookie_users_keep_existing_thread_isolation(self):
        first = self.register()
        first_id = first.json()["user_id"]
        self.login()
        self.assertEqual(
            self.invoke("shared-thread", self.csrf_headers()).status_code,
            200,
        )
        self.client.cookies.clear()
        second = self.register("bob", "bob@example.com")
        second_id = second.json()["user_id"]
        self.login("bob")
        denied = self.invoke("shared-thread", self.csrf_headers())
        self.assertEqual(denied.status_code, 403)
        self.assertNotEqual(first_id, second_id)

    def test_auth_routes_are_documented(self):
        paths = self.client.get("/openapi.json").json()["paths"]
        self.assertIn("/api/v1/auth/register", paths)
        self.assertIn("/api/v1/auth/login", paths)
        self.assertIn("/api/v1/auth/me", paths)
        self.assertIn("/api/v1/auth/logout", paths)


if __name__ == "__main__":
    unittest.main()
