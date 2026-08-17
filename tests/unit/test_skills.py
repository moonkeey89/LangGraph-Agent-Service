import unittest

from ai_agent_learning.skills import calculate, get_weather, search_attraction


class SkillTests(unittest.TestCase):
    def test_weather_uses_business_data(self):
        self.assertEqual(get_weather("上海"), "小雨，22℃")

    def test_weather_rejects_unsupported_city(self):
        self.assertIn("暂不支持", get_weather("深圳"))

    def test_attraction_uses_business_data(self):
        self.assertIn("八达岭长城", search_attraction("北京"))

    def test_calculator_supports_basic_arithmetic(self):
        cases = {
            "6 * 7": "42",
            "(10 + 2) / 3": "4.0",
            "2 ** 10": "1024",
            "-5 + 2": "-3",
        }

        for expression, expected in cases.items():
            with self.subTest(expression=expression):
                self.assertEqual(calculate(expression), expected)

    def test_calculator_rejects_python_code(self):
        expressions = [
            "__import__('os').getcwd()",
            "open('secret.txt')",
            "True + 1",
            "2 ** 1000",
        ]

        for expression in expressions:
            with self.subTest(expression=expression):
                self.assertEqual(calculate(expression), "无法计算")

    def test_calculator_handles_invalid_math(self):
        self.assertEqual(calculate("1 / 0"), "无法计算")
        self.assertEqual(calculate("not math"), "无法计算")


if __name__ == "__main__":
    unittest.main()
