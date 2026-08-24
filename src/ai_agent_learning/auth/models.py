from dataclasses import dataclass
from datetime import datetime, timezone
import re


MAX_USERNAME_LENGTH = 64
MAX_EMAIL_LENGTH = 254
MIN_PASSWORD_LENGTH = 8
MAX_PASSWORD_LENGTH = 128
_USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_EMAIL_PATTERN = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def to_utc_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def from_utc_text(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def normalize_username(value: str) -> str:
    normalized = value.strip().casefold()
    if not 3 <= len(normalized) <= MAX_USERNAME_LENGTH:
        raise ValueError("用户名长度必须在 3 到 64 个字符之间")
    if not _USERNAME_PATTERN.fullmatch(normalized):
        raise ValueError("用户名只能包含字母、数字、点、下划线和连字符")
    return normalized


def normalize_email(value: str) -> str:
    normalized = value.strip().casefold()
    if len(normalized) > MAX_EMAIL_LENGTH or not _EMAIL_PATTERN.fullmatch(normalized):
        raise ValueError("邮箱格式无效")
    return normalized


def validate_password(value: str) -> str:
    if not MIN_PASSWORD_LENGTH <= len(value) <= MAX_PASSWORD_LENGTH:
        raise ValueError("密码长度必须在 8 到 128 个字符之间")
    return value


@dataclass(frozen=True)
class User:
    user_id: str
    username: str
    email: str
    password_hash: str
    is_active: bool
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class Session:
    session_id: str
    user_id: str
    token_hash: str
    csrf_token_hash: str
    expires_at: str
    revoked_at: str | None
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class AuthenticatedSession:
    user: User
    session: Session


@dataclass(frozen=True)
class IssuedSession:
    user: User
    session: Session
    session_token: str
    csrf_token: str
