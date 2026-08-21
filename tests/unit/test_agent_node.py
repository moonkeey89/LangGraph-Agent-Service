import unittest

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

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
        self.assertIsInstance(llm.received_messages[0], SystemMessage)
        self.assertIs(llm.received_messages[1], messages[0])
        self.assertIn("ToolMessage", llm.received_messages[0].content)
        self.assertEqual(result["messages"][0].content, "测试回答")

    def test_node_injects_recalled_memories_without_user_id(self):
        llm = FakeChatModel()
        node = AgentNode(llm, TOOLS)

        node.run(
            {
                "messages": [HumanMessage(content="我喜欢什么")],
                "recalled_memories": [
                    {
                        "memory_id": "memory-1",
                        "content": "用户喜欢吃青菜",
                        "memory_type": "preference",
                    }
                ],
            }
        )

        system_message = llm.received_messages[0]
        self.assertIn("用户喜欢吃青菜", system_message.content)
        self.assertNotIn("user_001", system_message.content)


if __name__ == "__main__":
    unittest.main()
