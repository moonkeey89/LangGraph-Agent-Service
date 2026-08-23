import unittest

from langchain_core.messages import AIMessage

from ai_agent_learning.agents import (
    build_supervisor_graph,
    create_knowledge_agent,
    create_math_agent,
    create_travel_agent,
)
from ai_agent_learning.agents.subagent_tools import create_subagent_tools
from ai_agent_learning.knowledge.models import KnowledgeSearchResponse


class DirectBoundModel:
    def invoke(self, _messages):
        return AIMessage(content="完成")


class RecordingModel:
    def __init__(self):
        self.bound_tool_sets: list[frozenset[str]] = []

    def bind_tools(self, tools):
        names = frozenset(tool.name for tool in tools)
        self.bound_tool_sets.append(names)
        return DirectBoundModel()

    def invoke(self, _messages):
        return AIMessage(content="基于证据回答")


class EmptyKnowledgeRetriever:
    def search(self, **kwargs):
        return KnowledgeSearchResponse(
            status="no_evidence",
            knowledge_base_id=kwargs["knowledge_base_id"],
            results=[],
            message="未找到可靠证据",
        )


class SubagentTests(unittest.TestCase):
    def test_specialists_bind_only_allowed_tools(self):
        llm = RecordingModel()
        travel = create_travel_agent(llm)
        math = create_math_agent(llm)

        self.assertEqual(
            travel.tool_names,
            {"get_weather", "search_attraction"},
        )
        self.assertEqual(math.tool_names, {"calculate"})
        self.assertNotIn("calculate", travel.tool_names)
        self.assertNotIn("get_weather", math.tool_names)

    def test_handoff_tool_schema_exposes_task_but_not_context_ids(self):
        llm = RecordingModel()
        tools = create_subagent_tools(
            create_travel_agent(llm),
            create_math_agent(llm),
        )

        self.assertEqual(
            {tool.name for tool in tools},
            {"ask_travel_agent", "ask_math_agent"},
        )
        for tool in tools:
            schema = tool.tool_call_schema.model_json_schema()
            properties = schema.get("properties", {})
            self.assertEqual(set(properties), {"task"})
            self.assertNotIn("user_id", properties)
            self.assertNotIn("thread_id", properties)

    def test_knowledge_agent_has_only_controlled_search_tool(self):
        llm = RecordingModel()
        knowledge = create_knowledge_agent(
            llm,
            retriever=EmptyKnowledgeRetriever(),
            knowledge_base_id="demo",
            top_k=3,
        )
        tools = create_subagent_tools(
            create_travel_agent(llm),
            create_math_agent(llm),
            knowledge_agent=knowledge,
        )

        self.assertEqual(knowledge.tool_names, {"search_knowledge_base"})
        schema = knowledge.search_tool.tool_call_schema.model_json_schema()
        self.assertEqual(set(schema["properties"]), {"query"})
        self.assertEqual(
            {tool.name for tool in tools},
            {"ask_travel_agent", "ask_math_agent", "ask_knowledge_agent"},
        )

    def test_supervisor_binds_handoffs_but_not_specialist_tools(self):
        llm = RecordingModel()

        build_supervisor_graph(llm)

        supervisor_tools = llm.bound_tool_sets[-1]
        self.assertIn("ask_travel_agent", supervisor_tools)
        self.assertIn("ask_math_agent", supervisor_tools)
        self.assertIn("save_memory", supervisor_tools)
        self.assertNotIn("get_weather", supervisor_tools)
        self.assertNotIn("search_attraction", supervisor_tools)
        self.assertNotIn("calculate", supervisor_tools)


if __name__ == "__main__":
    unittest.main()
