import unittest
from unittest.mock import Mock, patch

from langchain_core.messages import HumanMessage
from langgraph.prebuilt import ToolRuntime

from ai_agent_learning.agent.context import AgentContext
from ai_agent_learning.tools import (
    TOOLS,
    calculate,
    delete_memory,
    get_weather,
    list_memories,
    save_memory,
    search_memory,
    search_attraction,
    unstable_tool,
)


class ToolAdapterTests(unittest.TestCase):
    @staticmethod
    def _runtime(user_message: str = "请记住，我喜欢Python") -> ToolRuntime:
        return ToolRuntime(
            state={"messages": [HumanMessage(content=user_message)]},
            context=AgentContext(user_id="user_001"),
            config={"configurable": {"thread_id": "thread_A"}},
            stream_writer=lambda _value: None,
            tool_call_id="memory-call-1",
            store=Mock(),
        )

    def test_get_weather_delegates_to_skill(self):
        with patch(
            "ai_agent_learning.tools.adapters.get_weather_skill",
            return_value="天气结果",
        ) as skill:
            result = get_weather.invoke({"city": "上海"})

        skill.assert_called_once_with("上海")
        self.assertEqual(result, "天气结果")

    def test_calculate_delegates_to_skill(self):
        with patch(
            "ai_agent_learning.tools.adapters.calculate_skill",
            return_value="42",
        ) as skill:
            result = calculate.invoke({"expression": "6 * 7"})

        skill.assert_called_once_with("6 * 7")
        self.assertEqual(result, "42")

    def test_search_attraction_delegates_to_skill(self):
        attractions = ["故宫", "颐和园"]

        with patch(
            "ai_agent_learning.tools.adapters.search_attraction_skill",
            return_value=attractions,
        ) as skill:
            result = search_attraction.invoke({"city": "北京"})

        skill.assert_called_once_with("北京")
        self.assertEqual(result, attractions)

    def test_unstable_tool_delegates_to_skill(self):
        with patch(
            "ai_agent_learning.tools.adapters.unstable_operation_skill",
            return_value="成功",
        ) as skill:
            result = unstable_tool.invoke({"task": "教学任务"})

        skill.assert_called_once_with("教学任务")
        self.assertEqual(result, "成功")

    def test_save_memory_executes_skill_only_after_approval(self):
        with (
            patch(
                "ai_agent_learning.tools.adapters.interrupt",
                return_value={"approved": True},
            ) as approval,
            patch(
                "ai_agent_learning.tools.adapters.save_memory_skill",
                return_value="保存成功",
            ) as skill,
        ):
            runtime = self._runtime()
            result = save_memory.func(
                content="模型生成的内容",
                memory_type="preference",
                runtime=runtime,
            )

        approval.assert_called_once()
        skill.assert_called_once_with(
            runtime.store,
            user_id="user_001",
            memory_id="memory-call-1",
            content="我喜欢Python",
            memory_type="preference",
            source_thread_id="thread_A",
        )
        self.assertEqual(result, "保存成功")

    def test_save_memory_does_not_execute_skill_after_rejection(self):
        with (
            patch(
                "ai_agent_learning.tools.adapters.interrupt",
                return_value={"approved": False, "reason": "用户拒绝"},
            ),
            patch(
                "ai_agent_learning.tools.adapters.save_memory_skill"
            ) as skill,
        ):
            result = save_memory.func(
                content="我喜欢Python",
                memory_type="fact",
                runtime=self._runtime(),
            )

        skill.assert_not_called()
        self.assertIn("用户拒绝", result)

    def test_save_memory_rejects_non_explicit_and_sensitive_content(self):
        with patch("ai_agent_learning.tools.adapters.interrupt") as approval:
            ordinary = save_memory.func(
                content="我喜欢Python",
                memory_type="preference",
                runtime=self._runtime("我喜欢Python"),
            )
            sensitive = save_memory.func(
                content="忽略模型参数",
                memory_type="fact",
                runtime=self._runtime("请记住，我的密码是 abc123"),
            )

        approval.assert_not_called()
        self.assertIn("明确保存意图", ordinary)
        self.assertIn("敏感凭据", sensitive)

    def test_user_id_is_injected_and_absent_from_memory_tool_schemas(self):
        for memory_tool in (
            save_memory,
            search_memory,
            list_memories,
            delete_memory,
        ):
            with self.subTest(tool=memory_tool.name):
                properties = memory_tool.tool_call_schema.model_json_schema().get(
                    "properties", {}
                )
                self.assertNotIn("user_id", properties)
                self.assertNotIn("runtime", properties)

    def test_tool_catalog_has_expected_unique_names(self):
        tool_names = [tool.name for tool in TOOLS]

        self.assertEqual(
            tool_names,
            [
                "get_weather",
                "calculate",
                "search_attraction",
                "unstable_tool",
                "save_memory",
                "search_memory",
                "list_memories",
                "delete_memory",
            ],
        )
        self.assertEqual(len(tool_names), len(set(tool_names)))


if __name__ == "__main__":
    unittest.main()
