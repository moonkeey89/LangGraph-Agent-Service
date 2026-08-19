import unittest
from unittest.mock import patch

from ai_agent_learning.tools import (
    TOOLS,
    calculate,
    get_weather,
    save_memory,
    search_attraction,
)


class ToolAdapterTests(unittest.TestCase):
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
            result = save_memory.invoke({"content": "我喜欢Python"})

        approval.assert_called_once()
        skill.assert_called_once_with("我喜欢Python")
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
            result = save_memory.invoke({"content": "我喜欢Python"})

        skill.assert_not_called()
        self.assertIn("用户拒绝", result)

    def test_tool_catalog_has_expected_unique_names(self):
        tool_names = [tool.name for tool in TOOLS]

        self.assertEqual(
            tool_names,
            ["get_weather", "calculate", "search_attraction", "save_memory"],
        )
        self.assertEqual(len(tool_names), len(set(tool_names)))


if __name__ == "__main__":
    unittest.main()
