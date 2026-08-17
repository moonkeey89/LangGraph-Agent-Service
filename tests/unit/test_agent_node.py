import unittest

from langchain_core.messages import AIMessage, HumanMessage

from ai_agent_learning.agent.node import AgentNode
from ai_agent_learning.tools import TOOLS


class FakeChatModel:
    def __init__(self):
        self.bound_tools = None
        self.received_messages = None

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    def invoke(self, messages):
        self.received_messages = messages
        return AIMessage(content="测试回答")


class AgentNodeTests(unittest.TestCase):
    def test_node_binds_tools_and_invokes_messages(self):
        llm = FakeChatModel()
        node = AgentNode(llm, TOOLS)
        messages = [HumanMessage(content="你好")]

        result = node.run({"messages": messages})

        self.assertIs(llm.bound_tools, TOOLS)
        self.assertIs(llm.received_messages, messages)
        self.assertEqual(result["messages"][0].content, "测试回答")


if __name__ == "__main__":
    unittest.main()
