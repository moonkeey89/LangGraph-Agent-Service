import unittest

from ai_agent_learning.agent.memory_recall import is_memory_recall_query


class MemoryRecallUnitTests(unittest.TestCase):
    def test_personal_questions_trigger_recall(self):
        for message in (
            "我是谁",
            "我叫什么名字",
            "我喜欢什么",
            "我爱哪些食物",
            "我有什么特点",
            "我主要使用什么语言",
            "我的目标是什么",
            "你还记得我吗",
        ):
            with self.subTest(message=message):
                self.assertTrue(is_memory_recall_query(message))

    def test_one_off_tasks_do_not_trigger_recall(self):
        for message in ("北京天气怎么样", "计算100×20", "你好"):
            with self.subTest(message=message):
                self.assertFalse(is_memory_recall_query(message))


if __name__ == "__main__":
    unittest.main()
