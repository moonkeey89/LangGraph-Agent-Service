import unittest

from langgraph.store.memory import InMemoryStore

from ai_agent_learning.skills import (
    calculate,
    delete_memory,
    extract_explicit_memory,
    get_weather,
    list_memories,
    save_memory,
    search_memory,
    search_attraction,
    update_memory,
)
from ai_agent_learning.skills.memory import (
    ensure_memory_is_safe,
    MemoryPolicyError,
)
from ai_agent_learning.skills.unstable import (
    get_unstable_attempts,
    reset_unstable_tool,
    run_unstable_operation,
)
from tests.helpers import DeterministicTestEmbeddings, TEST_EMBEDDING_DIMENSIONS


class SkillTests(unittest.TestCase):
    def tearDown(self):
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

    def test_memory_skill_supports_user_scoped_crud(self):
        store = InMemoryStore(
            index={
                "dims": TEST_EMBEDDING_DIMENSIONS,
                "embed": DeterministicTestEmbeddings(),
                "fields": ["content"],
            }
        )
        saved = save_memory(
            store,
            user_id="user_001",
            memory_id="memory-1",
            content="我主要使用Python",
            memory_type="preference",
            source_thread_id="thread_A",
        )

        self.assertEqual(saved["source"], "user_explicit")
        self.assertEqual(saved["status"], "active")
        self.assertEqual(
            search_memory(store, user_id="user_001", query="编程语言")[0][
                "memory_id"
            ],
            "memory-1",
        )
        self.assertEqual(list_memories(store, user_id="user_002"), [])
        self.assertFalse(
            delete_memory(store, user_id="user_002", memory_id="memory-1")
        )
        self.assertIsNone(
            update_memory(
                store,
                user_id="user_002",
                memory_id="memory-1",
                content="越权修改",
                memory_type="fact",
                source_thread_id="thread_B",
            )
        )
        updated = update_memory(
            store,
            user_id="user_001",
            memory_id="memory-1",
            content="我现在主要使用Python和LangGraph",
            memory_type="preference",
            source_thread_id="thread_B",
        )
        self.assertIsNotNone(updated)
        self.assertEqual(updated["source"], "memory_manager")
        self.assertEqual(updated["source_thread_id"], "thread_B")
        self.assertTrue(
            delete_memory(store, user_id="user_001", memory_id="memory-1")
        )
        self.assertEqual(list_memories(store, user_id="user_001"), [])

    def test_memory_policy_requires_explicit_intent_and_rejects_secrets(self):
        self.assertEqual(
            extract_explicit_memory("请记住，我主要使用Python。"),
            "我主要使用Python",
        )
        with self.assertRaises(MemoryPolicyError):
            extract_explicit_memory("我主要使用Python")
        with self.assertRaises(MemoryPolicyError):
            extract_explicit_memory("不要记住，我主要使用Python")
        with self.assertRaises(MemoryPolicyError):
            ensure_memory_is_safe("我的 API Key 是 sk-abcdefghijklmnop")
        with self.assertRaises(MemoryPolicyError):
            ensure_memory_is_safe("我的API Key是sk-test-123456789")

    def test_semantic_search_is_limited_to_three_memories(self):
        store = InMemoryStore(
            index={
                "dims": TEST_EMBEDDING_DIMENSIONS,
                "embed": DeterministicTestEmbeddings(),
                "fields": ["content"],
            }
        )
        for index in range(4):
            save_memory(
                store,
                user_id="user_001",
                memory_id=f"memory-{index}",
                content=f"Python偏好{index}",
                memory_type="preference",
                source_thread_id="thread_A",
            )

        self.assertEqual(
            len(search_memory(store, user_id="user_001", query="Python")),
            3,
        )

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
