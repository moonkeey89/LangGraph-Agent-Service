import logging
from collections.abc import Sequence
from typing import Literal

from langchain_core.messages import AIMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.store.base import BaseStore

from ai_agent_learning.agent.context import AgentContext
from ai_agent_learning.agent.error_recovery import (
    build_failure_response,
    clear_tool_error,
    human_review,
    route_after_human_review,
    route_after_tools,
    tool_error_boundary,
)
from ai_agent_learning.agent.node import AgentNode
from ai_agent_learning.agent.memory_manager import (
    DEFAULT_MEMORY_CONFIDENCE_THRESHOLD,
    MemoryExecutorNode,
    MemoryManagerNode,
)
from ai_agent_learning.agent.memory_recall import MemoryRecallNode
from ai_agent_learning.agent.state import AgentState


logger = logging.getLogger(__name__)


def route_after_agent(
    state: AgentState,
) -> Literal["tools", "memory_manager"]:
    """Continue ReAct tool use or enrich memory after the final answer."""
    messages = state.get("messages", [])
    last_message = messages[-1] if messages else None
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return "memory_manager"


def build_graph(
    llm: BaseChatModel,
    tools: Sequence[BaseTool],
    checkpointer: BaseCheckpointSaver | None = None,
    store: BaseStore | None = None,
    memory_manager_llm: BaseChatModel | None = None,
    memory_confidence_threshold: float = DEFAULT_MEMORY_CONFIDENCE_THRESHOLD,
):
    agent = AgentNode(llm, tools)
    memory_manager = MemoryManagerNode(memory_manager_llm or llm)
    memory_executor = MemoryExecutorNode(memory_confidence_threshold)
    memory_recall = MemoryRecallNode()
    tool_node = ToolNode(
        tools,
        handle_tool_errors=False,
        wrap_tool_call=tool_error_boundary,
    )
    graph = StateGraph(AgentState, context_schema=AgentContext)

    graph.add_node("memory_recall", memory_recall.run)
    graph.add_node("agent", agent.run)
    graph.add_node("tools", tool_node)
    graph.add_node("tool_success", clear_tool_error)
    graph.add_node("human_review", human_review)
    graph.add_node("failure", build_failure_response)
    graph.add_node("memory_manager", memory_manager.run)
    graph.add_node("memory_executor", memory_executor.run)
    graph.set_entry_point("memory_recall")
    graph.add_edge("memory_recall", "agent")
    graph.add_conditional_edges(
        "agent",
        route_after_agent,
        {"tools": "tools", "memory_manager": "memory_manager"},
    )
    graph.add_conditional_edges(
        "tools",
        route_after_tools,
        {
            "retry": "tools",
            "agent_correction": "agent",
            "human_review": "human_review",
            "fail": "failure",
            "success": "tool_success",
        },
    )
    graph.add_edge("tool_success", "agent")
    graph.add_conditional_edges(
        "human_review",
        route_after_human_review,
        {"retry": "tools", "cancel": "memory_manager"},
    )
    graph.add_edge("failure", "memory_manager")
    graph.add_edge("memory_manager", "memory_executor")
    graph.add_edge("memory_executor", END)

    logger.info("Compiled LangGraph ReAct workflow")
    return graph.compile(checkpointer=checkpointer, store=store)
