import unittest

from ai_agent_learning.agent.memory_manager import (
    is_memory_candidate,
    MemoryDecision,
)


class MemoryManagerUnitTests(unittest.TestCase):
    def test_decision_schema_has_required_fields_but_no_user_id(self):
        schema = MemoryDecision.model_json_schema()
        properties = schema["properties"]

        self.assertEqual(
            set(properties),
            {
                "operation",
                "memory_type",
                "content",
                "target_memory_id",
                "confidence",
                "reason",
            },
        )
        self.assertNotIn("user_id", properties)
        self.assertEqual(set(schema["required"]), set(properties))

    def test_lightweight_candidate_gate(self):
        candidate_messages = [
            "我叫小明",
            "我喜欢Python",
            "我爱吃青菜，最喜欢数学",
            "我平时使用PyCharm开发",
            "我现在改用VS Code开发了",
            "我的目标是学习LangGraph",
            "忘记我使用VS Code这件事",
        ]
        none_messages = [
            "你好",
            "北京天气怎么样？",
            "帮我计算100×20",
            "我主要使用什么语言？",
            "我喜欢什么",
            "请记住，我喜欢Python",
        ]

        for message in candidate_messages:
            with self.subTest(message=message):
                self.assertTrue(is_memory_candidate(message))
        for message in none_messages:
            with self.subTest(message=message):
                self.assertFalse(is_memory_candidate(message))


if __name__ == "__main__":
    unittest.main()
