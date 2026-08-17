import unittest
from unittest.mock import patch

from tools import TOOLS, calculate, get_weather, search_attraction


class ToolAdapterTests(unittest.TestCase):
    def test_get_weather_delegates_to_skill(self):
        with patch("tools.get_weather_skill", return_value="天气结果") as skill:
            result = get_weather.invoke({"city": "上海"})

        skill.assert_called_once_with("上海")
        self.assertEqual(result, "天气结果")

    def test_calculate_delegates_to_skill(self):
        with patch("tools.calculate_skill", return_value="42") as skill:
            result = calculate.invoke({"expression": "6 * 7"})

        skill.assert_called_once_with("6 * 7")
        self.assertEqual(result, "42")

    def test_search_attraction_delegates_to_skill(self):
        attractions = ["故宫", "颐和园"]

        with patch(
            "tools.search_attraction_skill",
            return_value=attractions,
        ) as skill:
            result = search_attraction.invoke({"city": "北京"})

        skill.assert_called_once_with("北京")
        self.assertEqual(result, attractions)

    def test_tool_catalog_has_expected_unique_names(self):
        tool_names = [tool.name for tool in TOOLS]

        self.assertEqual(
            tool_names,
            ["get_weather", "calculate", "search_attraction"],
        )
        self.assertEqual(len(tool_names), len(set(tool_names)))


if __name__ == "__main__":
    unittest.main()
