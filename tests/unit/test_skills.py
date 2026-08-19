import unittest

from ai_agent_learning.skills import (
    calculate,
    get_weather,
    save_memory,
    search_attraction,
)
from ai_agent_learning.skills.memory import (
    clear_saved_memories,
    get_saved_memories,
)
from ai_agent_learning.skills.unstable import (
    get_unstable_attempts,
    reset_unstable_tool,
    run_unstable_operation,
)


class SkillTests(unittest.TestCase):
    def tearDown(self):
        clear_saved_memories()
        reset_unstable_tool()

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

    def test_simulated_memory_skill_has_a_real_in_process_side_effect(self):
        result = save_memory("我喜欢Python")

        self.assertEqual(get_saved_memories(), ("我喜欢Python",))
        self.assertIn("已保存", result)

    def test_unstable_skill_fails_twice_then_succeeds_and_can_reset(self):
        task = "教学任务"

        with self.assertRaises(TimeoutError):
            run_unstable_operation(task)
        with self.assertRaises(TimeoutError):
            run_unstable_operation(task)
        self.assertIn("第 3 次尝试", run_unstable_operation(task))
        self.assertEqual(get_unstable_attempts(task), 3)

        reset_unstable_tool()
        self.assertEqual(get_unstable_attempts(task), 0)


if __name__ == "__main__":
    unittest.main()
