import json
import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from ai_agent_learning.agent.error_recovery import (
    build_failure_response,
    classify_tool_error,
    clear_tool_error,
    route_after_tools,
    tool_error_boundary,
)
from ai_agent_learning.agent.node import AgentNode
from ai_agent_learning.agent.state import AgentState


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SubagentResult:
    """JSON-serializable result returned across the Supervisor boundary."""

    agent_name: str
    status: Literal["success", "failed"]
    result: str | None
    error: str | None
    retry_recommended: bool
    sources: list[dict[str, object]] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _knowledge_sources(messages: list) -> list[dict[str, object]]:
    sources: list[dict[str, object]] = []
    seen: set[str] = set()
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        if message.name != "search_knowledge_base":
            continue
        try:
            payload = json.loads(str(message.content))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        results = payload.get("results", []) if isinstance(payload, dict) else []
        for result in results:
            if not isinstance(result, dict):
                continue
            chunk_id = str(result.get("chunk_id", ""))
            if not chunk_id or chunk_id in seen:
                continue
            seen.add(chunk_id)
            sources.append(
                {
                    key: result.get(key)
                    for key in (
                        "source",
                        "page",
                        "document_id",
                        "chunk_id",
                        "score",
                    )
                }
            )
    return sources


def _route_after_subagent(state: AgentState) -> Literal["tools", "end"]:
    messages = state.get("messages", [])
    last_message = messages[-1] if messages else None
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return "end"


def _build_stateless_react_graph(
    llm: BaseChatModel,
    tools: Sequence[BaseTool],
    system_prompt: str,
):
    """Build a small read-only ReAct graph without checkpoint or memory."""
    graph = StateGraph(AgentState)
    graph.add_node(
        "agent",
        AgentNode(
            llm,
            tools,
            additional_system_prompt=system_prompt,
        ).run,
    )
    graph.add_node(
        "tools",
        ToolNode(
            tools,
            handle_tool_errors=False,
            wrap_tool_call=tool_error_boundary,
        ),
    )
    graph.add_node("tool_success", clear_tool_error)
    graph.add_node("failure", build_failure_response)
    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent",
        _route_after_subagent,
        {"tools": "tools", "end": END},
    )
    graph.add_conditional_edges(
        "tools",
        route_after_tools,
        {
            "retry": "tools",
            "agent_correction": "agent",
            # Stateless subagents cannot pause for a CLI approval. Exhausted
            # read-only retries therefore degrade to a structured failure.
            "human_review": "failure",
            "fail": "failure",
            "success": "tool_success",
        },
    )
    graph.add_edge("tool_success", "agent")
    graph.add_edge("failure", END)
    return graph.compile()


class StatelessReActSubagent:
    """A short-lived specialist that receives one task and returns one summary."""

    def __init__(
        self,
        *,
        agent_name: str,
        llm: BaseChatModel,
        tools: Sequence[BaseTool],
        system_prompt: str,
        recursion_limit: int = 16,
    ):
        self.agent_name = agent_name
        self.tools = tuple(tools)
        self.tool_names = frozenset(tool.name for tool in self.tools)
        self.recursion_limit = recursion_limit
        self._graph = _build_stateless_react_graph(
            llm,
            self.tools,
            system_prompt,
        )

    def invoke(self, task: str) -> SubagentResult:
        normalized_task = task.strip()
        if not normalized_task:
            return SubagentResult(
                agent_name=self.agent_name,
                status="failed",
                result=None,
                error="子任务不能为空",
                retry_recommended=False,
            )

        try:
            final_state = self._graph.invoke(
                {"messages": [HumanMessage(content=normalized_task)]},
                config={"recursion_limit": self.recursion_limit},
            )
        except Exception as error:
            error_type = classify_tool_error(error)
            logger.exception("%s crashed", self.agent_name)
            return SubagentResult(
                agent_name=self.agent_name,
                status="failed",
                result=None,
                error=f"{type(error).__name__}: {error}",
                retry_recommended=error_type == "transient",
            )

        if final_state.get("status") == "failed":
            return SubagentResult(
                agent_name=self.agent_name,
                status="failed",
                result=None,
                error=str(final_state.get("error") or "子任务执行失败"),
                retry_recommended=(
                    final_state.get("error_type") == "transient"
                ),
            )

        messages = final_state.get("messages", [])
        final_message = messages[-1] if messages else None
        if not isinstance(final_message, AIMessage) or final_message.tool_calls:
            return SubagentResult(
                agent_name=self.agent_name,
                status="failed",
                result=None,
                error="Subagent 未生成可用的最终摘要",
                retry_recommended=False,
            )

        return SubagentResult(
            agent_name=self.agent_name,
            status="success",
            result=str(final_message.content),
            error=None,
            retry_recommended=False,
            sources=_knowledge_sources(messages),
        )
