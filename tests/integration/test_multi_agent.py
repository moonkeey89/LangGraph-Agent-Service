import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from ai_agent_learning.agent import AgentContext
from ai_agent_learning.agents import SubagentResult, build_supervisor_graph
from ai_agent_learning.checkpoint import open_sqlite_checkpointer


def _tool_names(tools) -> frozenset[str]:
    return frozenset(tool.name for tool in tools)


class RoleAwareBoundModel:
    def __init__(self, tool_names: frozenset[str]):
        self.tool_names = tool_names

    def invoke(self, messages):
        if "ask_travel_agent" in self.tool_names:
            return self._supervisor(messages)
        if self.tool_names == {"get_weather", "search_attraction"}:
            return self._travel(messages)
        if self.tool_names == {"calculate"}:
            return self._math(messages)
        raise AssertionError(f"Unexpected tool boundary: {self.tool_names}")

    @staticmethod
    def _latest_human(messages) -> str:
        return str(
            next(
                message.content
                for message in reversed(messages)
                if isinstance(message, HumanMessage)
            )
        )

    def _supervisor(self, messages):
        request = self._latest_human(messages)
        handoffs = [
            message
            for message in messages
            if isinstance(message, ToolMessage)
            and message.name in {"ask_travel_agent", "ask_math_agent"}
        ]
        called = {message.name for message in handoffs}
        needs_travel = "天气" in request or "景点" in request
        needs_math = "预算" in request or "等于多少" in request

        if needs_travel and "ask_travel_agent" not in called:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "ask_travel_agent",
                        "args": {"task": "查询北京天气和主要景点"},
                        "id": "handoff-travel",
                        "type": "tool_call",
                    }
                ],
            )
        if needs_math and "ask_math_agent" not in called:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "ask_math_agent",
                        "args": {"task": "计算 500 * 3"},
                        "id": "handoff-math",
                        "type": "tool_call",
                    }
                ],
            )
        if handoffs:
            summaries = [json.loads(message.content) for message in handoffs]
            return AIMessage(
                content="Supervisor整合："
                + "；".join(
                    str(item.get("result") or item.get("error"))
                    for item in summaries
                )
            )
        return AIMessage(content="这是不需要工具的直接回答。")

    def _travel(self, messages):
        tool_messages = [
            message for message in messages if isinstance(message, ToolMessage)
        ]
        called = {message.name for message in tool_messages}
        if "get_weather" not in called:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_weather",
                        "args": {"city": "北京"},
                        "id": "weather-1",
                        "type": "tool_call",
                    }
                ],
            )
        if "search_attraction" not in called:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_attraction",
                        "args": {"city": "北京"},
                        "id": "attraction-1",
                        "type": "tool_call",
                    },
                ],
            )
        return AIMessage(
            content="旅游摘要：北京天气已查询；主要景点包括故宫和长城。"
        )

    def _math(self, messages):
        tool_message = next(
            (
                message
                for message in reversed(messages)
                if isinstance(message, ToolMessage)
            ),
            None,
        )
        if tool_message is None:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "calculate",
                        "args": {"expression": "500 * 3"},
                        "id": "calculate-1",
                        "type": "tool_call",
                    }
                ],
            )
        return AIMessage(content=f"表达式500*3，结果{tool_message.content}。")


class RoleAwareModel:
    def bind_tools(self, tools):
        return RoleAwareBoundModel(_tool_names(tools))


class StubSubagent:
    def __init__(self, agent_name: str, result: SubagentResult):
        self.agent_name = agent_name
        self.result = result
        self.tasks: list[str] = []

    def invoke(self, task: str) -> SubagentResult:
        self.tasks.append(task)
        return self.result


class RepeatingSupervisorBoundModel:
    def __init__(self):
        self.call_count = 0

    def invoke(self, _messages):
        self.call_count += 1
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "ask_math_agent",
                    "args": {"task": "计算 500 * 3"},
                    "id": f"repeat-call-{self.call_count}",
                    "type": "tool_call",
                }
            ],
        )


class RepeatingSupervisorModel:
    def bind_tools(self, tools):
        names = _tool_names(tools)
        if "ask_math_agent" in names:
            return RepeatingSupervisorBoundModel()
        return RoleAwareBoundModel(names)


class UniqueRepeatingSupervisorBoundModel:
    def __init__(self):
        self.call_count = 0

    def invoke(self, _messages):
        self.call_count += 1
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "ask_math_agent",
                    "args": {"task": f"计算 500 * {self.call_count}"},
                    "id": f"unique-call-{self.call_count}",
                    "type": "tool_call",
                }
            ],
        )


class UniqueRepeatingSupervisorModel:
    def bind_tools(self, tools):
        names = _tool_names(tools)
        if "ask_math_agent" in names:
            return UniqueRepeatingSupervisorBoundModel()
        return RoleAwareBoundModel(names)


class MultiAgentIntegrationTests(unittest.TestCase):
    def _invoke(self, app, text: str, thread_id: str = "multi-001"):
        return app.invoke(
            {"messages": [HumanMessage(content=text)]},
            config={"configurable": {"thread_id": thread_id}},
            context=AgentContext(user_id="user-001"),
        )

    def test_single_domain_and_composite_routing(self):
        app = build_supervisor_graph(RoleAwareModel())

        travel = self._invoke(
            app,
            "北京天气怎么样，有哪些主要景点？",
            "travel-only",
        )
        travel_tools = [
            message.name
            for message in travel["messages"]
            if isinstance(message, ToolMessage)
        ]
        self.assertEqual(travel_tools, ["ask_travel_agent"])
        self.assertIn("旅游摘要", travel["messages"][-1].content)

        math = self._invoke(app, "500×3等于多少？", "math-only")
        math_tools = [
            message.name
            for message in math["messages"]
            if isinstance(message, ToolMessage)
        ]
        self.assertEqual(math_tools, ["ask_math_agent"])
        self.assertIn("1500", math["messages"][-1].content)

        composite = self._invoke(
            app,
            "查询北京天气和主要景点，如果每天预算500元，计算3天总预算。",
            "composite",
        )
        composite_tools = [
            message.name
            for message in composite["messages"]
            if isinstance(message, ToolMessage)
        ]
        self.assertEqual(
            composite_tools,
            ["ask_travel_agent", "ask_math_agent"],
        )
        self.assertIn("旅游摘要", composite["messages"][-1].content)
        self.assertIn("1500", composite["messages"][-1].content)
        self.assertEqual(len(composite["subagent_calls"]), 2)

    def test_direct_answer_does_not_force_handoff(self):
        result = self._invoke(
            build_supervisor_graph(RoleAwareModel()),
            "请解释什么是单一职责",
            "direct",
        )
        self.assertFalse(
            any(
                isinstance(message, ToolMessage)
                for message in result["messages"]
            )
        )
        self.assertEqual(
            result["messages"][-1].content,
            "这是不需要工具的直接回答。",
        )

    def test_subagent_failure_is_data_supervisor_can_combine(self):
        travel = StubSubagent(
            "travel_agent",
            SubagentResult(
                agent_name="travel_agent",
                status="failed",
                result=None,
                error="天气服务暂时不可用",
                retry_recommended=True,
            ),
        )
        app = build_supervisor_graph(
            RoleAwareModel(),
            travel_agent=travel,
        )
        result = self._invoke(
            app,
            "查询北京天气和主要景点，如果每天预算500元，计算3天总预算。",
            "partial-failure",
        )

        self.assertIn("天气服务暂时不可用", result["messages"][-1].content)
        self.assertIn("1500", result["messages"][-1].content)
        self.assertEqual(len(travel.tasks), 1)

    def test_duplicate_handoff_stops_loop_and_is_checkpointed(self):
        successful_math = StubSubagent(
            "math_agent",
            SubagentResult(
                agent_name="math_agent",
                status="success",
                result="1500",
                error=None,
                retry_recommended=False,
            ),
        )
        unused_travel = StubSubagent(
            "travel_agent",
            SubagentResult(
                agent_name="travel_agent",
                status="success",
                result="unused",
                error=None,
                retry_recommended=False,
            ),
        )
        with TemporaryDirectory() as directory:
            database = Path(directory) / "checkpoints.sqlite"
            with open_sqlite_checkpointer(database) as checkpointer:
                app = build_supervisor_graph(
                    RepeatingSupervisorModel(),
                    checkpointer=checkpointer,
                    travel_agent=unused_travel,
                    math_agent=successful_math,
                    max_subagent_calls=4,
                )
                result = self._invoke(app, "重复测试", "loop-guard")
                snapshot = app.get_state(
                    {"configurable": {"thread_id": "loop-guard"}}
                )

        self.assertEqual(len(successful_math.tasks), 1)
        self.assertEqual(
            [record["status"] for record in result["subagent_calls"]],
            ["success", "blocked"],
        )
        self.assertEqual(snapshot.values["status"], "failed")
        self.assertIn("相同任务", snapshot.values["error"])

    def test_max_handoffs_stops_unique_repeated_coordination(self):
        math = StubSubagent(
            "math_agent",
            SubagentResult(
                agent_name="math_agent",
                status="success",
                result="完成",
                error=None,
                retry_recommended=False,
            ),
        )
        travel = StubSubagent(
            "travel_agent",
            SubagentResult(
                agent_name="travel_agent",
                status="success",
                result="unused",
                error=None,
                retry_recommended=False,
            ),
        )
        app = build_supervisor_graph(
            UniqueRepeatingSupervisorModel(),
            travel_agent=travel,
            math_agent=math,
            max_subagent_calls=2,
        )

        result = self._invoke(app, "调用上限测试", "limit-guard")

        self.assertEqual(len(math.tasks), 2)
        self.assertEqual(
            [record["status"] for record in result["subagent_calls"]],
            ["success", "success", "blocked"],
        )
        self.assertIn("达到上限", result["error"])

    def test_supervisor_main_thread_restores_and_manager_runs_once(self):
        with TemporaryDirectory() as directory:
            database = Path(directory) / "checkpoints.sqlite"
            config = {"configurable": {"thread_id": "restart-thread"}}
            context = AgentContext(user_id="user-001")
            with open_sqlite_checkpointer(database) as first_checkpointer:
                first_app = build_supervisor_graph(
                    RoleAwareModel(),
                    checkpointer=first_checkpointer,
                )
                self._invoke(
                    first_app,
                    "北京天气怎么样，有哪些主要景点？",
                    "restart-thread",
                )
                history = list(first_app.get_state_history(config))
                manager_completions = [
                    snapshot
                    for snapshot in history
                    if snapshot.next == ("memory_executor",)
                ]
                self.assertEqual(len(manager_completions), 1)

            with open_sqlite_checkpointer(database) as second_checkpointer:
                second_app = build_supervisor_graph(
                    RoleAwareModel(),
                    checkpointer=second_checkpointer,
                )
                restored = second_app.get_state(config)

        self.assertTrue(restored.values["messages"])
        self.assertEqual(len(restored.values["subagent_calls"]), 1)


if __name__ == "__main__":
    unittest.main()
