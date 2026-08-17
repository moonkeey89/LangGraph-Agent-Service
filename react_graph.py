from langgraph.graph import StateGraph

from langgraph.prebuilt import tools_condition

from state import AgentState

from agent_node import AgentNode

from tool_node import tool_node

from tools import TOOLS



graph = StateGraph(
    AgentState
)



agent = AgentNode(TOOLS)



graph.add_node(
    "agent",
    agent.run
)



graph.add_node(
    "tools",
    tool_node
)



graph.set_entry_point(
    "agent"
)



graph.add_conditional_edges(
    "agent",
    tools_condition
)



graph.add_edge(
    "tools",
    "agent"
)



app = graph.compile()
