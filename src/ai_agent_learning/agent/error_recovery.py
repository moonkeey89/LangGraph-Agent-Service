from collections.abc import Callable
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.errors import GraphBubbleUp
from langgraph.prebuilt.tool_node import ToolCallRequest, ToolInvocationError
from langgraph.types import Command, interrupt
from pydantic import ValidationError

from ai_agent_learning.agent.state import AgentState, ErrorType


DEFAULT_MAX_RETRIES = 3
SIDE_EFFECT_TOOL_NAMES = frozenset({"save_memory"})
TRANSIENT_EXCEPTION_NAMES = frozenset(
    {"RateLimitError", "APIConnectionError", "APITimeoutError"}
)
PERMISSION_EXCEPTION_NAMES = frozenset(
    {"AuthenticationError", "PermissionDeniedError"}
)


class PermanentToolError(RuntimeError):
    """A known permanent tool failure that must not be retried."""


class SideEffectUnknownError(RuntimeError):
    """The tool may have produced a side effect before failing."""


def classify_tool_error(error: Exception) -> ErrorType:
    """Classify an exception conservatively; unknown errors are permanent."""
    if isinstance(error, SideEffectUnknownError):
        return "side_effect_unknown"
    if isinstance(error, (TimeoutError, ConnectionError)):
        return "transient"
    if type(error).__name__ in TRANSIENT_EXCEPTION_NAMES:
        return "transient"
    if isinstance(error, (ToolInvocationError, ValidationError, TypeError, ValueError)):
        return "invalid_arguments"
    if isinstance(error, PermissionError):
        return "permission"
    if type(error).__name__ in PERMISSION_EXCEPTION_NAMES:
        return "permission"
    if isinstance(error, PermanentToolError):
        return "permanent"
    return "permanent"


def _state_dict(request: ToolCallRequest) -> dict[str, Any]:
    return request.state if isinstance(request.state, dict) else {}


def _tool_message_id(tool_call_id: str) -> str:
    return f"tool-result:{tool_call_id}"


def _error_command(
    request: ToolCallRequest,
    error_type: ErrorType,
    error_text: str,
) -> Command:
    state = _state_dict(request)
    current_retry_count = int(state.get("retry_count", 0))
    max_retries = max(1, int(state.get("max_retries", DEFAULT_MAX_RETRIES)))

    if error_type == "transient":
        retry_count = current_retry_count + 1
        status = "human_review" if retry_count >= max_retries else "retry"
        guidance = (
            f"临时错误，第 {retry_count}/{max_retries} 次执行失败。"
            if status == "retry"
            else f"临时错误已达到 {max_retries} 次执行上限，等待人工复核。"
        )
    elif error_type == "invalid_arguments":
        retry_count = 0
        status = "agent_correction"
        guidance = "参数无效，请 Agent 修改参数后再发起新的工具调用。"
    else:
        retry_count = current_retry_count
        status = "failed"
        guidance = "该错误不允许自动重试。"

    call = request.tool_call
    message = ToolMessage(
        content=(
            f"工具 {call['name']} 执行失败；error_type={error_type}；"
            f"error={error_text}；{guidance}"
        ),
        name=call["name"],
        tool_call_id=call["id"],
        id=_tool_message_id(call["id"]),
        status="error",
    )
    return Command(
        update={
            "messages": [message],
            "status": status,
            "error": error_text,
            "error_type": error_type,
            "failed_node": "tools",
            "retry_count": retry_count,
            "max_retries": max_retries,
        }
    )


def tool_error_boundary(
    request: ToolCallRequest,
    execute: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:
    """Execute one tool call and convert classified errors into state updates."""
    try:
        result = execute(request)
    except GraphBubbleUp:
        # interrupt() uses this control-flow exception and must reach LangGraph.
        raise
    except Exception as error:
        error_type = classify_tool_error(error)
        if (
            request.tool_call["name"] in SIDE_EFFECT_TOOL_NAMES
            and error_type not in {"invalid_arguments", "permission"}
        ):
            error_type = "side_effect_unknown"
        return _error_command(request, error_type, str(error))

    if isinstance(result, ToolMessage):
        if result.status == "error":
            return _error_command(
                request,
                "invalid_arguments",
                str(result.content),
            )
        return result.model_copy(
            update={"id": _tool_message_id(request.tool_call["id"])}
        )

    return result


def route_after_tools(state: AgentState) -> str:
    """Choose retry, correction, review, failure, or success."""
    messages = state.get("messages", [])
    latest_ai_index = next(
        (
            index
            for index in range(len(messages) - 1, -1, -1)
            for message in [messages[index]]
            if isinstance(message, AIMessage) and message.tool_calls
        ),
        None,
    )
    latest_ai_message = (
        messages[latest_ai_index] if latest_ai_index is not None else None
    )
    expected_call_ids = (
        {call["id"] for call in latest_ai_message.tool_calls}
        if latest_ai_message is not None
        else set()
    )
    tool_results = (
        [
            message
            for message in messages[latest_ai_index + 1 :]
            if isinstance(message, ToolMessage)
            and message.tool_call_id in expected_call_ids
        ]
        if latest_ai_index is not None
        else []
    )
    completed_call_ids = {message.tool_call_id for message in tool_results}
    if (
        expected_call_ids
        and completed_call_ids == expected_call_ids
        and all(message.status != "error" for message in tool_results)
    ):
        return "success"

    status = state.get("status")
    if status in {"retry", "agent_correction", "human_review"}:
        return status
    return "fail"


def clear_tool_error(_state: AgentState) -> dict[str, Any]:
    """Clear the completed attempt's error metadata before returning to AgentNode."""
    return {
        "status": "success",
        "error": None,
        "error_type": None,
        "failed_node": None,
        "retry_count": 0,
    }


def human_review(state: AgentState) -> dict[str, Any]:
    """Pause after automatic retries are exhausted."""
    decision = interrupt(
        {
            "action": "tool_failure_review",
            "failed_node": state.get("failed_node", "tools"),
            "error": state.get("error", "未知错误"),
            "error_type": state.get("error_type", "transient"),
            "retry_count": state.get("retry_count", 0),
            "max_retries": state.get("max_retries", DEFAULT_MAX_RETRIES),
            "options": ["retry", "cancel"],
            "message": "自动重试已达到上限，请选择 retry 或 cancel。",
        }
    )

    if isinstance(decision, dict) and decision.get("action") == "retry":
        return {"status": "retry"}

    reason = (
        decision.get("reason", "用户取消")
        if isinstance(decision, dict)
        else "未收到有效人工决定"
    )
    return {
        "status": "cancelled",
        "messages": [
            AIMessage(
                content=(
                    "工具执行已取消，未再进行重试。"
                    f"失败原因：{state.get('error', '未知错误')}；{reason}"
                )
            )
        ],
    }


def route_after_human_review(state: AgentState) -> str:
    return "retry" if state.get("status") == "retry" else "cancel"


def build_failure_response(state: AgentState) -> dict[str, Any]:
    """Create a deterministic final answer for non-retryable failures."""
    return {
        "status": "failed",
        "messages": [
            AIMessage(
                content=(
                    "工具执行失败，系统未进行自动重试。"
                    f"错误类型：{state.get('error_type', 'permanent')}；"
                    f"原因：{state.get('error', '未知错误')}"
                )
            )
        ],
    }
