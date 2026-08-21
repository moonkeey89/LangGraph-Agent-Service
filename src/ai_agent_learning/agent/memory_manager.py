import json
import logging
import re
from hashlib import sha256
from typing import Any, Literal

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.runtime import Runtime
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ai_agent_learning.agent.context import AgentContext
from ai_agent_learning.agent.state import AgentState
from ai_agent_learning.skills.memory import (
    delete_memory,
    ensure_memory_is_safe,
    get_memory,
    list_memories,
    MAX_MEMORY_LENGTH,
    MemoryPolicyError,
    MemoryType,
    save_memory,
    search_memory,
    update_memory,
)


logger = logging.getLogger(__name__)

MemoryOperation = Literal["ADD", "UPDATE", "DELETE", "NONE"]
DEFAULT_MEMORY_CONFIDENCE_THRESHOLD = 0.75
MEMORY_SIDE_EFFECT_TOOLS = frozenset({"save_memory", "delete_memory"})

_EXPLICIT_SAVE_PATTERN = re.compile(
    r"(?:请帮我记住|请记住|帮我记住|记住这件事|记住)", re.IGNORECASE
)
_CANDIDATE_PATTERN = re.compile(
    r"(?:"
    r"我叫|我的名字(?:是|叫)|我喜欢|我不喜欢|我爱|最喜欢|"
    r"我(?:平时|主要|现在)?使用|我正在|我现在改用|"
    r"以后请|我的目标|忘记|忘掉|请忘记"
    r")",
    re.IGNORECASE,
)
_QUESTION_PATTERN = re.compile(
    r"(?:什么|谁|哪些|哪一个|多少|怎么|如何|吗|呢)[？?]?$",
    re.IGNORECASE,
)
_LOW_VALUE_PATTERN = re.compile(
    r"(?:天气|气温|几点|现在时间|帮我计算|计算一下|算一下|"
    r"景点|旅游景点)",
    re.IGNORECASE,
)
_GREETINGS = frozenset(
    {"你好", "您好", "嗨", "hello", "hi", "谢谢", "多谢", "再见"}
)

_DECISION_PROMPT = """你是长期记忆决策器，只能根据最新用户消息和候选旧记忆做决定。
允许的 operation：
- ADD：稳定、有跨会话价值且候选中不存在的新事实。
- UPDATE：新事实明确替代某条候选旧记忆，必须填写该候选的 memory_id。
- DELETE：用户明确要求忘记某条候选记忆，必须填写该候选的 memory_id。
- NONE：一次性任务、寒暄、问题、重复事实、模型推测或无长期价值内容。

只提取简洁、独立的用户事实；不要保存完整对话，不要根据AI回答推测事实。
target_memory_id 只能从候选记忆中选择。没有可靠目标时必须选择 NONE。
memory_type 只能是 preference、profile、fact、instruction、other。
所有字段都必须返回；NONE 时 content 使用空字符串、target_memory_id 使用 null。
不要输出 user_id。"""


class MemoryDecision(BaseModel):
    """Structured LLM decision. User identity is deliberately absent."""

    model_config = ConfigDict(extra="forbid")

    operation: MemoryOperation
    memory_type: MemoryType
    content: str = Field(max_length=MAX_MEMORY_LENGTH)
    target_memory_id: str | None
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1, max_length=500)

    @classmethod
    def none(cls, reason: str) -> "MemoryDecision":
        return cls(
            operation="NONE",
            memory_type="other",
            content="",
            target_memory_id=None,
            confidence=1.0,
            reason=reason,
        )


def _normalize_content(content: str) -> str:
    return "".join(content.casefold().split()).strip("。.!！")


def _latest_human_message(state: AgentState) -> tuple[int, str] | None:
    messages = state.get("messages", [])
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if isinstance(message, HumanMessage):
            return index, str(message.content).strip()
    return None


def _memory_operation_was_finally_handled(
    state: AgentState,
    human_index: int,
) -> bool:
    """Skip only final memory outcomes, not a non-explicit Tool refusal."""
    turn_messages = state.get("messages", [])[human_index + 1 :]
    memory_call_ids = {
        str(call.get("id", ""))
        for message in turn_messages
        if isinstance(message, AIMessage)
        for call in message.tool_calls
        if str(call.get("name", "")) in MEMORY_SIDE_EFFECT_TOOLS
    }
    if not memory_call_ids:
        return False

    tool_results = [
        message
        for message in turn_messages
        if isinstance(message, ToolMessage)
        and message.tool_call_id in memory_call_ids
    ]
    if not tool_results:
        # An unresolved tool/interrupt must never be bypassed by automatic memory.
        return True

    non_explicit_refusal = "未检测到“请记住”等明确保存意图"
    return any(
        non_explicit_refusal not in str(message.content)
        for message in tool_results
    )


def is_memory_candidate(user_message: str) -> bool:
    """Cheap gate that avoids an extra LLM call for obvious NONE messages."""
    text = user_message.strip()
    normalized = text.casefold().strip("。.!！")
    if not text or normalized in _GREETINGS:
        return False
    if _EXPLICIT_SAVE_PATTERN.search(text):
        # Explicit saves remain owned by save_memory + interrupt approval.
        return False
    if (
        ("?" in text or "？" in text or _QUESTION_PATTERN.search(text))
        and not re.search(r"忘记|忘掉", text)
    ):
        return False
    if _CANDIDATE_PATTERN.search(text):
        return True
    if _LOW_VALUE_PATTERN.search(text):
        return False
    return False


def _trusted_user_id(runtime: Runtime[AgentContext]) -> str:
    context = runtime.context
    user_id = context.user_id if isinstance(context, AgentContext) else ""
    if not isinstance(user_id, str) or not user_id.strip():
        raise MemoryPolicyError("运行时上下文缺少可信 user_id")
    return user_id.strip()


def _thread_id(runtime: Runtime[AgentContext]) -> str:
    if runtime.execution_info and runtime.execution_info.thread_id:
        return runtime.execution_info.thread_id
    return "unknown"


def _state_update(
    decision: MemoryDecision,
    *,
    candidate_ids: list[str] | None = None,
    status: str,
    error: str | None = None,
) -> dict[str, object]:
    return {
        "memory_decision": decision.model_dump(mode="json"),
        "memory_candidate_ids": candidate_ids or [],
        "memory_manager_status": status,
        "memory_manager_error": error,
    }


class MemoryManagerNode:
    """Read user-scoped candidates and produce a decision without writing."""

    def __init__(self, llm: Any):
        self.llm = llm
        self._decision_llm = None

    def _structured_llm(self):
        if self._decision_llm is None:
            self._decision_llm = self.llm.with_structured_output(
                MemoryDecision,
                method="function_calling",
            )
        return self._decision_llm

    def run(
        self,
        state: AgentState,
        runtime: Runtime[AgentContext],
    ) -> dict[str, object]:
        latest = _latest_human_message(state)
        if state.get("status") != "completed" or latest is None:
            return _state_update(
                MemoryDecision.none("本轮没有可分析的正常最终回答"),
                status="skipped",
            )

        human_index, user_message = latest
        if _memory_operation_was_finally_handled(state, human_index):
            return _state_update(
                MemoryDecision.none("本轮已由显式记忆工具处理"),
                status="skipped",
            )
        if not is_memory_candidate(user_message):
            return _state_update(
                MemoryDecision.none("轻量规则判定为无长期记忆价值"),
                status="skipped",
            )
        if runtime.store is None:
            return _state_update(
                MemoryDecision.none("长期记忆 Store 未配置"),
                status="skipped",
            )

        try:
            # Sensitive user content must not be sent to the decision model.
            ensure_memory_is_safe(user_message)
            user_id = _trusted_user_id(runtime)
            candidates = search_memory(
                runtime.store,
                user_id=user_id,
                query=user_message,
            )
            candidate_payload = [
                {
                    "memory_id": memory["memory_id"],
                    "content": memory["content"],
                    "memory_type": memory["memory_type"],
                }
                for memory in candidates
            ]
            candidate_ids = [item["memory_id"] for item in candidate_payload]
            model_input = {
                "latest_user_message": user_message,
                "candidate_memories": candidate_payload,
            }
            raw_decision = self._structured_llm().invoke(
                [
                    SystemMessage(content=_DECISION_PROMPT),
                    HumanMessage(
                        content=json.dumps(model_input, ensure_ascii=False)
                    ),
                ]
            )
            decision = (
                raw_decision
                if isinstance(raw_decision, MemoryDecision)
                else MemoryDecision.model_validate(raw_decision)
            )

            if decision.operation == "ADD":
                normalized = _normalize_content(decision.content)
                if any(
                    _normalize_content(memory["content"]) == normalized
                    for memory in candidates
                ):
                    decision = MemoryDecision.none("候选记忆中已存在相同事实")

            return _state_update(
                decision,
                candidate_ids=candidate_ids,
                status="decided",
            )
        except MemoryPolicyError as error:
            logger.warning("Memory Manager policy rejected input: %s", error)
            return _state_update(
                MemoryDecision.none(f"Memory Policy 否决：{error}"),
                status="rejected",
                error=str(error),
            )
        except Exception as error:
            # Memory enrichment is optional; never replace a valid Agent answer.
            logger.exception("Memory Manager decision failed")
            return _state_update(
                MemoryDecision.none("Memory Manager 失败，安全降级为 NONE"),
                status="failed",
                error=f"{type(error).__name__}: {error}",
            )


class MemoryExecutorNode:
    """Validate an untrusted decision in code, then call memory Skills."""

    def __init__(
        self,
        confidence_threshold: float = DEFAULT_MEMORY_CONFIDENCE_THRESHOLD,
    ):
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("Memory Manager confidence threshold 必须在 0 到 1 之间")
        self.confidence_threshold = confidence_threshold

    def run(
        self,
        state: AgentState,
        runtime: Runtime[AgentContext],
    ) -> dict[str, object]:
        try:
            decision = MemoryDecision.model_validate(
                state.get("memory_decision")
                or MemoryDecision.none("缺少记忆决定").model_dump()
            )
            if decision.operation == "NONE":
                return _state_update(
                    decision,
                    candidate_ids=state.get("memory_candidate_ids", []),
                    status=state.get("memory_manager_status", "skipped"),
                    error=state.get("memory_manager_error"),
                )
            if runtime.store is None:
                raise MemoryPolicyError("长期记忆 Store 未配置")

            user_id = _trusted_user_id(runtime)
            if decision.confidence < self.confidence_threshold:
                raise MemoryPolicyError(
                    f"置信度 {decision.confidence:.2f} 低于阈值 "
                    f"{self.confidence_threshold:.2f}"
                )

            active_memories = list_memories(runtime.store, user_id=user_id)
            candidate_ids = set(state.get("memory_candidate_ids", []))
            thread_id = _thread_id(runtime)

            if decision.operation in {"ADD", "UPDATE"}:
                content = decision.content.strip()
                if not content:
                    raise MemoryPolicyError("ADD/UPDATE 的 content 不能为空")
                ensure_memory_is_safe(content)

            if decision.operation == "ADD":
                normalized = _normalize_content(decision.content)
                if any(
                    _normalize_content(memory["content"]) == normalized
                    for memory in active_memories
                ):
                    raise MemoryPolicyError("已存在内容完全相同的有效记忆")
                memory_id = self._memory_id(
                    user_id, thread_id, user_message=_latest_human_message(state)
                )
                save_memory(
                    runtime.store,
                    user_id=user_id,
                    memory_id=memory_id,
                    content=decision.content.strip(),
                    memory_type=decision.memory_type,
                    source_thread_id=thread_id,
                    source="memory_manager",
                )
            else:
                target_id = (decision.target_memory_id or "").strip()
                if not target_id:
                    raise MemoryPolicyError("UPDATE/DELETE 缺少 target_memory_id")
                if target_id not in candidate_ids:
                    raise MemoryPolicyError(
                        "target_memory_id 不属于本次提供给模型的候选记忆"
                    )
                if get_memory(
                    runtime.store,
                    user_id=user_id,
                    memory_id=target_id,
                ) is None:
                    raise MemoryPolicyError(
                        "target_memory_id 不存在或不属于当前用户"
                    )

                if decision.operation == "UPDATE":
                    updated = update_memory(
                        runtime.store,
                        user_id=user_id,
                        memory_id=target_id,
                        content=decision.content.strip(),
                        memory_type=decision.memory_type,
                        source_thread_id=thread_id,
                    )
                    if updated is None:
                        raise MemoryPolicyError("目标记忆更新失败")
                elif not delete_memory(
                    runtime.store,
                    user_id=user_id,
                    memory_id=target_id,
                ):
                    raise MemoryPolicyError("目标记忆删除失败")

            return _state_update(
                decision,
                candidate_ids=list(candidate_ids),
                status="applied",
            )
        except (MemoryPolicyError, ValidationError, ValueError) as error:
            logger.warning("Memory Policy rejected decision: %s", error)
            return _state_update(
                MemoryDecision.none(f"Memory Policy 否决：{error}"),
                status="rejected",
                error=str(error),
            )
        except Exception as error:
            logger.exception("Memory Executor failed")
            return _state_update(
                MemoryDecision.none("Memory Executor 失败，未执行记忆操作"),
                status="failed",
                error=f"{type(error).__name__}: {error}",
            )

    @staticmethod
    def _memory_id(
        user_id: str,
        thread_id: str,
        *,
        user_message: tuple[int, str] | None,
    ) -> str:
        message_text = user_message[1] if user_message else ""
        digest = sha256(
            f"{user_id}\0{thread_id}\0{message_text}".encode()
        ).hexdigest()
        return f"managed-memory-{digest[:24]}"
