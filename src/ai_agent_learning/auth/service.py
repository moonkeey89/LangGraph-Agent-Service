from dataclasses import dataclass
from datetime import timedelta
import hashlib
import hmac
import secrets
import sqlite3
import uuid

from pwdlib import PasswordHash

from ai_agent_learning.auth.catalog import AuthCatalog
from ai_agent_learning.auth.models import (
    AuthenticatedSession,
    IssuedSession,
    Session,
    User,
    from_utc_text,
    normalize_email,
    normalize_username,
    to_utc_text,
    utc_now,
    validate_password,
)


class AuthServiceError(RuntimeError):
    status_code = 400
    public_message = "Authentication request could not be completed"


class AuthConflictError(AuthServiceError):
    status_code = 409
    public_message = "Username or email is already registered"


class InvalidCredentialsError(AuthServiceError):
    status_code = 401
    public_message = "Invalid username/email or password"


class AuthenticationRequiredError(AuthServiceError):
    status_code = 401
    public_message = "Authentication required"


class CsrfValidationError(AuthServiceError):
    status_code = 403
    public_message = "CSRF validation failed"


@dataclass(frozen=True)
class AuthCookieConfig:
    session_cookie_name: str
    csrf_cookie_name: str
    csrf_header_name: str
    secure: bool
    same_site: str
    domain: str | None
    session_ttl_minutes: int

    @property
    def max_age_seconds(self) -> int:
        return self.session_ttl_minutes * 60


class AuthService:
    def __init__(self, catalog: AuthCatalog, cookie_config: AuthCookieConfig):
        self.catalog = catalog
        self.cookie_config = cookie_config
        self.password_hash = PasswordHash.recommended()
        # Verify this hash for unknown accounts to reduce username enumeration
        # through large timing differences. It is never persisted.
        self._dummy_password_hash = self.password_hash.hash(
            secrets.token_urlsafe(24)
        )

    def register(self, *, username: str, email: str, password: str) -> User:
        normalized_username = normalize_username(username)
        normalized_email = normalize_email(email)
        validate_password(password)
        now = to_utc_text(utc_now())
        user = User(
            user_id=f"usr_{uuid.uuid4().hex}",
            username=normalized_username,
            email=normalized_email,
            password_hash=self.password_hash.hash(password),
            is_active=True,
            created_at=now,
            updated_at=now,
        )
        try:
            return self.catalog.create_user(user)
        except sqlite3.IntegrityError as error:
            raise AuthConflictError from error

    def login(self, *, login: str, password: str) -> IssuedSession:
        normalized_login = login.strip().casefold()
        user = (
            self.catalog.get_user_by_email(normalized_login)
            if "@" in normalized_login
            else self.catalog.get_user_by_username(normalized_login)
        )
        password_hash = user.password_hash if user is not None else self._dummy_password_hash
        verified = self.password_hash.verify(password, password_hash)
        if user is None or not verified or not user.is_active:
            raise InvalidCredentialsError

        session_token = secrets.token_urlsafe(32)
        csrf_token = secrets.token_urlsafe(32)
        now_value = utc_now()
        now = to_utc_text(now_value)
        session = Session(
            session_id=f"ses_{uuid.uuid4().hex}",
            user_id=user.user_id,
            token_hash=_token_hash(session_token),
            csrf_token_hash=_token_hash(csrf_token),
            expires_at=to_utc_text(
                now_value + timedelta(minutes=self.cookie_config.session_ttl_minutes)
            ),
            revoked_at=None,
            created_at=now,
            updated_at=now,
        )
        self.catalog.create_session(session)
        return IssuedSession(
            user=user,
            session=session,
            session_token=session_token,
            csrf_token=csrf_token,
        )

    def authenticate(self, session_token: str | None) -> AuthenticatedSession:
        if not session_token:
            raise AuthenticationRequiredError
        session = self.catalog.get_session_by_token_hash(_token_hash(session_token))
        if session is None or session.revoked_at is not None:
            raise AuthenticationRequiredError
        if from_utc_text(session.expires_at) <= utc_now():
            self.catalog.revoke_session(session.session_id, to_utc_text(utc_now()))
            raise AuthenticationRequiredError
        user = self.catalog.get_user_by_id(session.user_id)
        if user is None or not user.is_active:
            raise AuthenticationRequiredError
        return AuthenticatedSession(user=user, session=session)

    def validate_csrf(
        self,
        authenticated: AuthenticatedSession,
        *,
        csrf_cookie: str | None,
        csrf_header: str | None,
    ) -> None:
        if not csrf_cookie or not csrf_header:
            raise CsrfValidationError
        if not hmac.compare_digest(csrf_cookie, csrf_header):
            raise CsrfValidationError
        if not hmac.compare_digest(
            _token_hash(csrf_cookie),
            authenticated.session.csrf_token_hash,
        ):
            raise CsrfValidationError

    def logout(self, authenticated: AuthenticatedSession) -> None:
        self.catalog.revoke_session(
            authenticated.session.session_id,
            to_utc_text(utc_now()),
        )


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()
