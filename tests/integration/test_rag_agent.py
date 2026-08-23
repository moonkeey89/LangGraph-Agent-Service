import json
import unittest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from ai_agent_learning.agent import AgentContext
from ai_agent_learning.agents import CriticDecision, SubagentResult, build_supervisor_graph
from ai_agent_learning.api.service import AgentService


class KnowledgeRoutingBoundModel:
    def __init__(self, tool_names):
        self.tool_names = frozenset(tool_names)

    def invoke(self, messages):
        request = next(
            str(message.content)
            for message in reversed(messages)
            if isinstance(message, HumanMessage)
        )
        handoffs = [
            message
            for message in messages
            if isinstance(message, ToolMessage)
            and message.name.startswith("ask_")
        ]
        called = {message.name for message in handoffs}
        if "手册" in request and "ask_knowledge_agent" not in called:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "ask_knowledge_agent",
                        "args": {"task": "根据手册回答项目代号"},
                        "id": "knowledge-1",
                        "type": "tool_call",
                    }
                ],
            )
        if "天气" in request and "ask_travel_agent" not in called:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "ask_travel_agent",
                        "args": {"task": "查询北京天气"},
                        "id": "travel-1",
                        "type": "tool_call",
                    }
                ],
            )
        if handoffs:
            payload = json.loads(handoffs[-1].content)
            return AIMessage(content=str(payload.get("result") or payload.get("error")))
        return AIMessage(content="普通问题直接回答")


class PassingCritic:
    def invoke(self, _messages):
        return CriticDecision(
            verdict="PASS",
            issues=[],
            suggestions=[],
            severity="none",
            reason="引用来自实际检索结果",
        )


class RoutingModel:
    def bind_tools(self, tools):
        return KnowledgeRoutingBoundModel(tool.name for tool in tools)

    def with_structured_output(self, _schema, method=None):
        return PassingCritic()


class StubSubagent:
    def __init__(self, agent_name, result):
        self.agent_name = agent_name
        self.result = result
        self.tasks = []

    def invoke(self, task):
        self.tasks.append(task)
        return self.result


class RagAgentIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.knowledge = StubSubagent(
            "knowledge_agent",
            SubagentResult(
                agent_name="knowledge_agent",
                status="success",
                result="项目代号是ORBIT-731。来源：manual.md。",
                error=None,
                retry_recommended=False,
                sources=[
                    {
                        "source": "manual.md",
                        "page": 2,
                        "document_id": "doc-real",
                        "chunk_id": "chunk-real",
                        "score": 0.96,
                    }
                ],
            ),
        )
        self.travel = StubSubagent(
            "travel_agent",
            SubagentResult(
                agent_name="travel_agent",
                status="success",
                result="北京晴",
                error=None,
                retry_recommended=False,
            ),
        )
        self.app = build_supervisor_graph(
            RoutingModel(),
            knowledge_agent=self.knowledge,
            travel_agent=self.travel,
        )

    def invoke(self, message, thread_id):
        return self.app.invoke(
            {"messages": [HumanMessage(content=message)]},
            config={"configurable": {"thread_id": thread_id}},
            context=AgentContext(user_id="rag-user"),
        )

    def test_private_document_routes_to_knowledge_and_sources_are_real(self):
        result = self.invoke("根据内部手册，项目代号是什么？", "rag-knowledge")
        tool_messages = [
            item for item in result["messages"] if isinstance(item, ToolMessage)
        ]
        self.assertEqual([item.name for item in tool_messages], ["ask_knowledge_agent"])
        self.assertEqual(len(self.knowledge.tasks), 1)
        response = AgentService._to_result(result, "rag-knowledge")
        self.assertEqual(response.sources[0].chunk_id, "chunk-real")
        self.assertEqual(response.sources[0].source, "manual.md")

    def test_travel_and_plain_questions_do_not_force_knowledge_agent(self):
        travel = self.invoke("北京天气怎么样？", "rag-travel")
        plain = self.invoke("解释单一职责", "rag-plain")
        self.assertTrue(
            any(
                isinstance(item, ToolMessage) and item.name == "ask_travel_agent"
                for item in travel["messages"]
            )
        )
        self.assertFalse(
            any(isinstance(item, ToolMessage) for item in plain["messages"])
        )
        self.assertEqual(self.knowledge.tasks, [])


if __name__ == "__main__":
    unittest.main()
