import logging
from collections.abc import Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from ai_agent_learning.agent.error_recovery import (
    build_failure_response,
    clear_tool_error,
    human_review,
    route_after_human_review,
    route_after_tools,
    tool_error_boundary,
)
from ai_agent_learning.agent.node import AgentNode
from ai_agent_learning.agent.state import AgentState


logger = logging.getLogger(__name__)


def build_graph(
    llm: BaseChatModel,
    tools: Sequence[BaseTool],
    checkpointer: BaseCheckpointSaver | None = None,
):
    agent = AgentNode(llm, tools)
    tool_node = ToolNode(
        tools,
        handle_tool_errors=False,
        wrap_tool_call=tool_error_boundary,
    )
    graph = StateGraph(AgentState)

    graph.add_node("agent", agent.run)
    graph.add_node("tools", tool_node)
    graph.add_node("tool_success", clear_tool_error)
    graph.add_node("human_review", human_review)
    graph.add_node("failure", build_failure_response)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", tools_condition)
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
        {"retry": "tools", "cancel": END},
    )
    graph.add_edge("failure", END)

    logger.info("Compiled LangGraph ReAct workflow")
    return graph.compile(checkpointer=checkpointer)
