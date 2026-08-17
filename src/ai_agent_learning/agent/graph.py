import logging
from collections.abc import Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from ai_agent_learning.agent.node import AgentNode
from ai_agent_learning.agent.state import AgentState


logger = logging.getLogger(__name__)


def build_graph(llm: BaseChatModel, tools: Sequence[BaseTool]):
    agent = AgentNode(llm, tools)
    tool_node = ToolNode(tools)
    graph = StateGraph(AgentState)

    graph.add_node("agent", agent.run)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")

    logger.info("Compiled LangGraph ReAct workflow")
    return graph.compile()
