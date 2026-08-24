import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ai_agent_learning.auth import (
    AuthCatalog,
    AuthConflictError,
    AuthCookieConfig,
    AuthenticationRequiredError,
    AuthService,
    InvalidCredentialsError,
)


def cookie_config() -> AuthCookieConfig:
    return AuthCookieConfig(
        session_cookie_name="test_session",
        csrf_cookie_name="test_csrf",
        csrf_header_name="X-CSRF-Token",
        secure=False,
        same_site="lax",
        domain=None,
        session_ttl_minutes=60,
    )


class AuthCatalogAndServiceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.path = Path(self.temporary.name) / "auth.sqlite"
        self.catalog = AuthCatalog(self.path)
        self.service = AuthService(self.catalog, cookie_config())

    def tearDown(self):
        self.catalog.close()
        self.temporary.cleanup()

    def test_schema_is_created_idempotently_in_temporary_database(self):
        tables = {
            row[0]
            for row in self.catalog.connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        self.assertIn("users", tables)
        self.assertIn("sessions", tables)
        self.assertIn("auth_schema_migrations", tables)
        self.catalog.close()
        self.catalog = AuthCatalog(self.path)
        version = self.catalog.connection.execute(
            "SELECT MAX(version) FROM auth_schema_migrations"
        ).fetchone()[0]
        self.assertEqual(version, 1)

    def test_registration_is_unique_and_password_is_argon2_hash(self):
        user = self.service.register(
            username="Researcher",
            email="Researcher@Example.com",
            password="correct-password",
        )
        stored = self.catalog.get_user_by_id(user.user_id)
        self.assertEqual(stored.username, "researcher")
        self.assertEqual(stored.email, "researcher@example.com")
        self.assertNotEqual(stored.password_hash, "correct-password")
        self.assertTrue(stored.password_hash.startswith("$argon2"))
        with self.assertRaises(AuthConflictError):
            self.service.register(
                username="researcher",
                email="other@example.com",
                password="another-password",
            )
        with self.assertRaises(AuthConflictError):
            self.service.register(
                username="another",
                email="RESEARCHER@example.com",
                password="another-password",
            )

    def test_login_verifies_password_and_uses_generic_failure(self):
        self.service.register(
            username="researcher",
            email="researcher@example.com",
            password="correct-password",
        )
        issued = self.service.login(
            login="researcher",
            password="correct-password",
        )
        self.assertEqual(issued.user.username, "researcher")
        for login, password in (
            ("researcher", "wrong-password"),
            ("missing", "wrong-password"),
        ):
            with self.subTest(login=login), self.assertRaises(
                InvalidCredentialsError
            ) as caught:
                self.service.login(login=login, password=password)
            self.assertEqual(
                caught.exception.public_message,
                "Invalid username/email or password",
            )

    def test_expired_revoked_and_inactive_sessions_are_rejected(self):
        user = self.service.register(
            username="researcher",
            email="researcher@example.com",
            password="correct-password",
        )
        expired = self.service.login(
            login="researcher", password="correct-password"
        )
        with self.catalog.connection:
            self.catalog.connection.execute(
                "UPDATE sessions SET expires_at = ? WHERE session_id = ?",
                ("2000-01-01T00:00:00+00:00", expired.session.session_id),
            )
        with self.assertRaises(AuthenticationRequiredError):
            self.service.authenticate(expired.session_token)

        revoked = self.service.login(
            login="researcher", password="correct-password"
        )
        authenticated = self.service.authenticate(revoked.session_token)
        self.service.logout(authenticated)
        with self.assertRaises(AuthenticationRequiredError):
            self.service.authenticate(revoked.session_token)

        inactive = self.service.login(
            login="researcher", password="correct-password"
        )
        with self.catalog.connection:
            self.catalog.connection.execute(
                "UPDATE users SET is_active = 0 WHERE user_id = ?",
                (user.user_id,),
            )
        with self.assertRaises(AuthenticationRequiredError):
            self.service.authenticate(inactive.session_token)

    def test_database_never_stores_raw_session_or_csrf_token(self):
        self.service.register(
            username="researcher",
            email="researcher@example.com",
            password="correct-password",
        )
        issued = self.service.login(
            login="researcher", password="correct-password"
        )
        row = self.catalog.connection.execute(
            "SELECT token_hash, csrf_token_hash FROM sessions WHERE session_id = ?",
            (issued.session.session_id,),
        ).fetchone()
        self.assertNotEqual(row[0], issued.session_token)
        self.assertNotEqual(row[1], issued.csrf_token)
        self.assertEqual(len(row[0]), 64)
        self.assertEqual(len(row[1]), 64)

    def test_user_and_session_survive_catalog_restart(self):
        self.service.register(
            username="researcher",
            email="researcher@example.com",
            password="correct-password",
        )
        issued = self.service.login(
            login="researcher", password="correct-password"
        )
        self.catalog.close()
        self.catalog = AuthCatalog(self.path)
        self.service = AuthService(self.catalog, cookie_config())
        restored = self.service.authenticate(issued.session_token)
        self.assertEqual(restored.user.user_id, issued.user.user_id)


if __name__ == "__main__":
    unittest.main()
