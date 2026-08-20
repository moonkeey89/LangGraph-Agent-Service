"""Explicit, user-scoped long-term memory business capability."""

from datetime import UTC, datetime
import re
from typing import Any, Literal, Protocol, TypedDict


MemoryType = Literal["preference", "profile", "fact", "instruction", "other"]
MemoryStatus = Literal["active", "deleted"]


class MemoryStore(Protocol):
    """Minimal storage port required by the memory business capability."""

    def get(self, namespace: tuple[str, ...], key: str) -> Any: ...

    def put(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
        index: bool | list[str] | None = None,
    ) -> None: ...

    def search(
        self,
        namespace_prefix: tuple[str, ...],
        *,
        query: str | None = None,
        filter: dict[str, Any] | None = None,
        limit: int = 10,
    ) -> list[Any]: ...


class MemoryRecord(TypedDict):
    memory_id: str
    content: str
    user_id: str
    memory_type: MemoryType
    source: Literal["user_explicit"]
    source_thread_id: str
    created_at: str
    updated_at: str
    status: MemoryStatus


MEMORY_NAMESPACE_ROOT = ("ai_agent_learning", "users")
MAX_MEMORY_LENGTH = 300
MAX_LIST_RESULTS = 20
SEARCH_TOP_K = 3
EXPLICIT_MEMORY_PATTERN = re.compile(
    r"(?:^|[。！？!?]\s*)(?:请帮我记住|请记住|帮我记住|记住这件事|记住)"
    r"\s*[：:,，]?\s*(?P<content>.+)",
    re.IGNORECASE | re.DOTALL,
)
NEGATIVE_MEMORY_PATTERN = re.compile(
    r"(?:不要|不用|别)(?:再)?(?:记住|保存)|请忘记|忘掉",
    re.IGNORECASE,
)
SENSITIVE_PATTERNS = (
    re.compile(r"\bapi[\s_-]*key\b", re.IGNORECASE),
    re.compile(r"\baccess[\s_-]*token\b", re.IGNORECASE),
    re.compile(r"\brefresh[\s_-]*token\b", re.IGNORECASE),
    re.compile(r"\bpassword\b", re.IGNORECASE),
    re.compile(r"\bverification[\s_-]*code\b", re.IGNORECASE),
    re.compile(r"\botp\b", re.IGNORECASE),
    re.compile(r"\bbearer\s+[a-z0-9._~+/=-]+", re.IGNORECASE),
    re.compile(r"\bsk-[a-z0-9_-]{12,}\b", re.IGNORECASE),
    re.compile(r"\bgh[pousr]_[a-z0-9]{12,}\b", re.IGNORECASE),
    re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"密码|口令|验证码|访问令牌|密钥|私钥"),
)


class MemoryPolicyError(ValueError):
    """The requested memory violates an explicit storage policy."""


def memory_namespace(user_id: str) -> tuple[str, ...]:
    normalized_user_id = user_id.strip()
    if not normalized_user_id:
        raise MemoryPolicyError("user_id 不能为空")
    return (*MEMORY_NAMESPACE_ROOT, normalized_user_id, "memories")


def extract_explicit_memory(user_message: str) -> str:
    """Extract a short standalone fact only from an explicit memory request."""
    if NEGATIVE_MEMORY_PATTERN.search(user_message):
        raise MemoryPolicyError("检测到否定保存意图")
    match = EXPLICIT_MEMORY_PATTERN.search(user_message.strip())
    if match is None:
        raise MemoryPolicyError("未检测到“请记住”等明确保存意图")

    content = " ".join(match.group("content").strip().split())
    content = content.strip("。.!！")
    if not content:
        raise MemoryPolicyError("明确保存请求中没有可保存的事实")
    if len(content) > MAX_MEMORY_LENGTH:
        raise MemoryPolicyError(
            f"记忆正文不能超过 {MAX_MEMORY_LENGTH} 个字符，请只保存简洁事实"
        )
    return content


def ensure_memory_is_safe(content: str) -> None:
    if any(pattern.search(content) for pattern in SENSITIVE_PATTERNS):
        raise MemoryPolicyError(
            "检测到 API Key、密码、验证码、令牌或其他敏感凭据，已拒绝保存"
        )


def save_memory(
    store: MemoryStore,
    *,
    user_id: str,
    memory_id: str,
    content: str,
    memory_type: MemoryType,
    source_thread_id: str,
) -> MemoryRecord:
    """Persist one approved, explicit user memory."""
    ensure_memory_is_safe(content)
    now = datetime.now(UTC).isoformat()
    namespace = memory_namespace(user_id)
    existing = store.get(namespace, memory_id)
    created_at = (
        str(existing.value.get("created_at", now)) if existing is not None else now
    )
    record: MemoryRecord = {
        "memory_id": memory_id,
        "content": content,
        "user_id": user_id,
        "memory_type": memory_type,
        "source": "user_explicit",
        "source_thread_id": source_thread_id,
        "created_at": created_at,
        "updated_at": now,
        "status": "active",
    }
    store.put(namespace, memory_id, record)
    return record


def search_memory(
    store: MemoryStore,
    *,
    user_id: str,
    query: str,
) -> list[MemoryRecord]:
    """Semantically search only active memories in one user's namespace."""
    results = store.search(
        memory_namespace(user_id),
        query=query.strip(),
        filter={"status": "active"},
        limit=SEARCH_TOP_K,
    )
    return [result.value for result in results]


def list_memories(
    store: MemoryStore,
    *,
    user_id: str,
) -> list[MemoryRecord]:
    """List a bounded number of active memories for one user."""
    results = store.search(
        memory_namespace(user_id),
        filter={"status": "active"},
        limit=MAX_LIST_RESULTS,
    )
    return [result.value for result in results]


def delete_memory(
    store: MemoryStore,
    *,
    user_id: str,
    memory_id: str,
) -> bool:
    """Soft-delete a memory only inside the caller's user namespace."""
    namespace = memory_namespace(user_id)
    item = store.get(namespace, memory_id)
    if item is None or item.value.get("status") != "active":
        return False

    updated = dict(item.value)
    updated["status"] = "deleted"
    updated["updated_at"] = datetime.now(UTC).isoformat()
    store.put(namespace, memory_id, updated, index=False)
    return True
