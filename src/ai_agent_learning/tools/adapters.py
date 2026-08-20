from hashlib import sha256
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import interrupt

from ai_agent_learning.agent.context import AgentContext
from ai_agent_learning.agent.state import AgentState
from ai_agent_learning.skills import (
    calculate as calculate_skill,
    delete_memory as delete_memory_skill,
    extract_explicit_memory,
    get_weather as get_weather_skill,
    list_memories as list_memories_skill,
    MemoryPolicyError,
    MemoryType,
    save_memory as save_memory_skill,
    search_memory as search_memory_skill,
    search_attraction as search_attraction_skill,
    run_unstable_operation as unstable_operation_skill,
)


@tool
def get_weather(city: str) -> str:
    """查询指定城市的天气。"""

    return get_weather_skill(city)


@tool
def calculate(expression: str) -> str:
    """计算给定的数学表达式。"""

    return calculate_skill(expression)


@tool
def search_attraction(city: str) -> list[str] | str:
    """查询指定城市的旅游景点。"""

    return search_attraction_skill(city)


@tool
def unstable_tool(task: str) -> str:
    """执行无副作用的教学任务；临时超时时由Graph进行有限重试。"""

    return unstable_operation_skill(task)


@tool
def save_memory(
    content: str,
    runtime: ToolRuntime[AgentContext, AgentState],
    memory_type: MemoryType = "fact",
) -> dict[str, Any] | str:
    """仅保存用户用“请记住”等措辞明确要求长期记住的简洁事实。"""
    if runtime.store is None:
        return "长期记忆 Store 未配置，无法保存。"

    user_message = _latest_human_message(runtime)
    try:
        # The human message is authoritative. The LLM-provided content is not trusted
        # as evidence that the user explicitly requested storage.
        explicit_content = extract_explicit_memory(user_message)
    except MemoryPolicyError as error:
        return f"拒绝保存长期记忆：{error}"

    # Run the sensitive-data check before asking for approval, but perform no write.
    try:
        from ai_agent_learning.skills.memory import ensure_memory_is_safe

        ensure_memory_is_safe(explicit_content)
    except MemoryPolicyError as error:
        return f"拒绝保存长期记忆：{error}"

    decision = interrupt(
        {
            "action": "save_user_memory",
            "tool_name": "save_memory",
            "arguments": {
                "content": explicit_content,
                "memory_type": memory_type,
            },
            "message": "该操作将写入一条长期记忆，是否批准？",
        }
    )

    if not isinstance(decision, dict) or decision.get("approved") is not True:
        reason = (
            decision.get("reason", "未获得明确批准")
            if isinstance(decision, dict)
            else "未获得明确批准"
        )
        return f"保存操作已取消：{reason}"

    user_id = _trusted_user_id(runtime)
    source_thread_id = _thread_id(runtime)
    memory_id = runtime.tool_call_id or _fallback_memory_id(
        user_id, source_thread_id, explicit_content
    )
    return save_memory_skill(
        runtime.store,
        user_id=user_id,
        memory_id=memory_id,
        content=explicit_content,
        memory_type=memory_type,
        source_thread_id=source_thread_id,
    )


@tool
def search_memory(
    query: str,
    runtime: ToolRuntime[AgentContext, AgentState],
) -> list[dict[str, Any]] | str:
    """语义检索当前用户明确保存的长期记忆，最多返回3条。"""
    if runtime.store is None:
        return "长期记忆 Store 未配置，无法检索。"
    return search_memory_skill(
        runtime.store,
        user_id=_trusted_user_id(runtime),
        query=query,
    )


@tool
def list_memories(
    runtime: ToolRuntime[AgentContext, AgentState],
) -> list[dict[str, Any]] | str:
    """列出当前用户已保存且仍有效的长期记忆。"""
    if runtime.store is None:
        return "长期记忆 Store 未配置，无法列出。"
    return list_memories_skill(
        runtime.store,
        user_id=_trusted_user_id(runtime),
    )


@tool
def delete_memory(
    memory_id: str,
    runtime: ToolRuntime[AgentContext, AgentState],
) -> str:
    """删除当前用户自己的一条长期记忆；执行前必须获得人工批准。"""
    if runtime.store is None:
        return "长期记忆 Store 未配置，无法删除。"

    decision = interrupt(
        {
            "action": "delete_user_memory",
            "tool_name": "delete_memory",
            "arguments": {"memory_id": memory_id},
            "message": "该操作将删除一条长期记忆，是否批准？",
        }
    )
    if not isinstance(decision, dict) or decision.get("approved") is not True:
        reason = (
            decision.get("reason", "未获得明确批准")
            if isinstance(decision, dict)
            else "未获得明确批准"
        )
        return f"删除操作已取消：{reason}"

    deleted = delete_memory_skill(
        runtime.store,
        user_id=_trusted_user_id(runtime),
        memory_id=memory_id,
    )
    return "长期记忆已删除。" if deleted else "未找到当前用户的对应长期记忆。"


def _latest_human_message(
    runtime: ToolRuntime[AgentContext, AgentState],
) -> str:
    state = runtime.state if isinstance(runtime.state, dict) else {}
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


def _trusted_user_id(runtime: ToolRuntime[AgentContext, AgentState]) -> str:
    context = runtime.context
    if isinstance(context, AgentContext):
        user_id = context.user_id
    elif isinstance(context, dict):
        user_id = context.get("user_id", "")
    else:
        user_id = ""
    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("运行时上下文缺少可信 user_id")
    return user_id.strip()


def _thread_id(runtime: ToolRuntime[AgentContext, AgentState]) -> str:
    configurable = runtime.config.get("configurable", {})
    return str(configurable.get("thread_id", "unknown"))


def _fallback_memory_id(user_id: str, thread_id: str, content: str) -> str:
    digest = sha256(f"{user_id}\0{thread_id}\0{content}".encode()).hexdigest()
    return f"memory-{digest[:24]}"


TOOLS = [
    get_weather,
    calculate,
    search_attraction,
    unstable_tool,
    save_memory,
    search_memory,
    list_memories,
    delete_memory,
]
