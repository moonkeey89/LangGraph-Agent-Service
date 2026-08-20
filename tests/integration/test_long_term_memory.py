import re
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from ai_agent_learning.agent import AgentContext, build_graph
from ai_agent_learning.checkpoint import open_sqlite_checkpointer
from ai_agent_learning.memory_store import open_sqlite_memory_store
from ai_agent_learning.skills.memory import list_memories as list_memories_skill
from ai_agent_learning.tools import TOOLS
from tests.helpers import DeterministicTestEmbeddings, TEST_EMBEDDING_DIMENSIONS


class MemoryCrudModel:
    """Deterministic Tool Calling model for memory integration tests."""

    def bind_tools(self, _tools):
        return self

    def invoke(self, messages):
        last_message = messages[-1]
        if isinstance(last_message, ToolMessage):
            content = str(last_message.content)
            if last_message.name == "search_memory":
                answer = "Python" if "Python" in content else "未找到相关长期记忆"
                return AIMessage(content=answer)
            if last_message.name == "delete_memory":
                return AIMessage(content=content)
            return AIMessage(content="长期记忆保存成功。")

        human_text = str(last_message.content)
        if "请记住" in human_text or "记住这件事" in human_text:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "save_memory",
                        "args": {
                            "content": "模型参数不作为保存依据",
                            "memory_type": "preference",
                        },
                        "id": "language-memory-1",
                        "type": "tool_call",
                    }
                ],
            )
        if "删除" in human_text:
            memory_id = re.search(r"memory_id=([^\s]+)", human_text).group(1)
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "delete_memory",
                        "args": {"memory_id": memory_id},
                        "id": f"delete-{memory_id}",
                        "type": "tool_call",
                    }
                ],
            )
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_memory",
                    "args": {"query": "主要使用的编程语言"},
                    "id": "search-language-1",
                    "type": "tool_call",
                }
            ],
        )


class AlwaysSaveModel:
    """Deliberately misbehaving model used to verify the policy boundary."""

    def bind_tools(self, _tools):
        return self

    def invoke(self, messages):
        if isinstance(messages[-1], ToolMessage):
            return AIMessage(content=str(messages[-1].content))
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "save_memory",
                    "args": {"content": "模型试图保存", "memory_type": "fact"},
                    "id": "policy-memory-1",
                    "type": "tool_call",
                }
            ],
        )


class LongTermMemoryTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = TemporaryDirectory()
        root = Path(self.temporary_directory.name)
        self.checkpoint_path = root / "checkpoints.sqlite"
        self.memory_path = root / "memories.sqlite"

    def tearDown(self):
        self.temporary_directory.cleanup()

    @staticmethod
    def _config(thread_id: str) -> dict:
        return {"configurable": {"thread_id": thread_id}}

    def _open_store(self):
        return open_sqlite_memory_store(
            self.memory_path,
            embeddings=DeterministicTestEmbeddings(),
            dimensions=TEST_EMBEDDING_DIMENSIONS,
        )

    def _save_language_memory(self, app, *, thread_id: str, user_id: str):
        config = self._config(thread_id)
        context = AgentContext(user_id=user_id)
        interrupted = app.invoke(
            {
                "messages": [
                    HumanMessage(content="请记住，我主要使用Python。")
                ]
            },
            config=config,
            context=context,
        )
        self.assertTrue(interrupted["__interrupt__"])
        return app.invoke(
            Command(resume={"approved": True}),
            config=config,
            context=context,
        )

    def test_cross_thread_user_isolation_restart_and_checkpoint_independence(self):
        user_001 = AgentContext(user_id="user_001")

        with (
            open_sqlite_checkpointer(self.checkpoint_path) as checkpointer,
            self._open_store() as store,
        ):
            first_app = build_graph(
                MemoryCrudModel(), TOOLS, checkpointer=checkpointer, store=store
            )
            self._save_language_memory(
                first_app, thread_id="thread_A", user_id="user_001"
            )

        # Both SQLite connections are reopened here, simulating a full restart.
        with (
            open_sqlite_checkpointer(self.checkpoint_path) as checkpointer,
            self._open_store() as store,
        ):
            restarted_app = build_graph(
                MemoryCrudModel(), TOOLS, checkpointer=checkpointer, store=store
            )
            thread_b = self._config("thread_B")
            self.assertEqual(restarted_app.get_state(thread_b).values, {})

            shared_result = restarted_app.invoke(
                {
                    "messages": [
                        HumanMessage(content="我主要使用什么编程语言？")
                    ]
                },
                config=thread_b,
                context=user_001,
            )
            isolated_result = restarted_app.invoke(
                {
                    "messages": [
                        HumanMessage(content="我主要使用什么编程语言？")
                    ]
                },
                config=self._config("thread_C"),
                context=AgentContext(user_id="user_002"),
            )

            self.assertEqual(shared_result["messages"][-1].content, "Python")
            self.assertEqual(
                isolated_result["messages"][-1].content,
                "未找到相关长期记忆",
            )
            self.assertFalse(
                any(
                    isinstance(message, HumanMessage)
                    and "请记住" in str(message.content)
                    for message in shared_result["messages"]
                )
            )

    def test_non_explicit_and_sensitive_requests_are_not_saved(self):
        cases = [
            ("北京天气怎么样？", "明确保存意图"),
            ("请记住，我的 API Key 是 sk-abcdefghijklmnop", "敏感凭据"),
        ]

        with (
            open_sqlite_checkpointer(self.checkpoint_path) as checkpointer,
            self._open_store() as store,
        ):
            app = build_graph(
                AlwaysSaveModel(), TOOLS, checkpointer=checkpointer, store=store
            )
            for index, (message, expected) in enumerate(cases):
                with self.subTest(message=message):
                    result = app.invoke(
                        {"messages": [HumanMessage(content=message)]},
                        config=self._config(f"policy_{index}"),
                        context=AgentContext(user_id="user_001"),
                    )
                    self.assertNotIn("__interrupt__", result)
                    self.assertIn(expected, result["messages"][-1].content)

            self.assertEqual(
                list_memories_skill(store, user_id="user_001"), []
            )

    def test_delete_is_user_scoped_and_requires_approval(self):
        with (
            open_sqlite_checkpointer(self.checkpoint_path) as checkpointer,
            self._open_store() as store,
        ):
            app = build_graph(
                MemoryCrudModel(), TOOLS, checkpointer=checkpointer, store=store
            )
            self._save_language_memory(
                app, thread_id="save_A", user_id="user_001"
            )

            wrong_user_config = self._config("delete_wrong_user")
            wrong_user_context = AgentContext(user_id="user_002")
            interrupted = app.invoke(
                {
                    "messages": [
                        HumanMessage(
                            content="删除 memory_id=language-memory-1"
                        )
                    ]
                },
                config=wrong_user_config,
                context=wrong_user_context,
            )
            self.assertTrue(interrupted["__interrupt__"])
            wrong_user_result = app.invoke(
                Command(resume={"approved": True}),
                config=wrong_user_config,
                context=wrong_user_context,
            )
            self.assertIn("未找到", wrong_user_result["messages"][-1].content)
            self.assertEqual(
                len(list_memories_skill(store, user_id="user_001")), 1
            )

            owner_config = self._config("delete_owner")
            owner_context = AgentContext(user_id="user_001")
            app.invoke(
                {
                    "messages": [
                        HumanMessage(
                            content="删除 memory_id=language-memory-1"
                        )
                    ]
                },
                config=owner_config,
                context=owner_context,
            )
            deleted = app.invoke(
                Command(resume={"approved": True}),
                config=owner_config,
                context=owner_context,
            )
            self.assertIn("已删除", deleted["messages"][-1].content)
            self.assertEqual(
                list_memories_skill(store, user_id="user_001"), []
            )


if __name__ == "__main__":
    unittest.main()
