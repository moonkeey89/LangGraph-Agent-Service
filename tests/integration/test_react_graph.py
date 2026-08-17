import unittest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from ai_agent_learning.agent import build_graph
from ai_agent_learning.tools import TOOLS


class DirectAnswerModel:
    def bind_tools(self, _tools):
        return self

    def invoke(self, _messages):
        return AIMessage(content="直接回答")


class ToolCallingModel:
    def __init__(self):
        self.invocation_count = 0

    def bind_tools(self, _tools):
        return self

    def invoke(self, messages):
        self.invocation_count += 1

        if self.invocation_count == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "calculate",
                        "args": {"expression": "6 * 7"},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            )

        tool_message = next(
            message for message in messages if isinstance(message, ToolMessage)
        )
        return AIMessage(content=f"计算结果是 {tool_message.content}")


class ReactGraphTests(unittest.TestCase):
    def test_graph_structure_is_unchanged(self):
        app = build_graph(DirectAnswerModel(), TOOLS)
        graph = app.get_graph()
        nodes = set(graph.nodes)
        edges = {(edge.source, edge.target) for edge in graph.edges}

        self.assertEqual(nodes, {"__start__", "agent", "tools", "__end__"})
        self.assertTrue(
            {
                ("__start__", "agent"),
                ("agent", "tools"),
                ("agent", "__end__"),
                ("tools", "agent"),
            }.issubset(edges)
        )

    def test_graph_can_end_without_tool_call(self):
        app = build_graph(DirectAnswerModel(), TOOLS)

        result = app.invoke(
            {"messages": [HumanMessage(content="你好")]}
        )

        self.assertEqual(result["messages"][-1].content, "直接回答")

    def test_graph_executes_tool_and_returns_to_agent(self):
        llm = ToolCallingModel()
        app = build_graph(llm, TOOLS)

        result = app.invoke(
            {"messages": [HumanMessage(content="计算 6 * 7")]}
        )

        self.assertEqual(llm.invocation_count, 2)
        self.assertTrue(
            any(isinstance(message, ToolMessage) for message in result["messages"])
        )
        self.assertEqual(result["messages"][-1].content, "计算结果是 42")


if __name__ == "__main__":
    unittest.main()
