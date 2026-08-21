import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from ai_agent_learning.agent import AgentContext, build_graph
from ai_agent_learning.checkpoint import open_sqlite_checkpointer
from ai_agent_learning.skills.memory import (
    list_memories as list_memories_skill,
)
from ai_agent_learning.memory_store import open_sqlite_memory_store
from ai_agent_learning.tools import TOOLS
from tests.helpers import DeterministicTestEmbeddings, TEST_EMBEDDING_DIMENSIONS


class DirectAnswerModel:
    def bind_tools(self, _tools):
        return self

    def invoke(self, _messages):
        return AIMessage(content="直接回答")


class ToolCallingModel:
    def __init__(self):
        self.invocation_count = 0

    def bind_tools(self, _tools):
        return self

    def invoke(self, messages):
        self.invocation_count += 1

        if self.invocation_count == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "calculate",
                        "args": {"expression": "6 * 7"},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            )

        tool_message = next(
            message for message in messages if isinstance(message, ToolMessage)
        )
        return AIMessage(content=f"计算结果是 {tool_message.content}")


class NameMemoryModel:
    def bind_tools(self, _tools):
        return self

    def invoke(self, messages):
        last_human_message = next(
            message
            for message in reversed(messages)
            if isinstance(message, HumanMessage)
        )

        if "我的名字是小明" in last_human_message.content:
            return AIMessage(content="好的，我记住了，你的名字是小明。")

        if "我叫什么名字" in last_human_message.content:
            knows_name = any(
                isinstance(message, HumanMessage)
                and "我的名字是小明" in message.content
                for message in messages[:-1]
            )
            if knows_name:
                return AIMessage(content="你的名字是小明。")

            return AIMessage(content="我不知道你的名字。")

        return AIMessage(content="好的。")


class SaveMemoryToolCallingModel:
    def bind_tools(self, _tools):
        return self

    def invoke(self, messages):
        last_tool_message = next(
            (
                message
                for message in reversed(messages)
                if isinstance(message, ToolMessage)
                and message.name == "save_memory"
            ),
            None,
        )
        if last_tool_message is not None:
            if "取消" in last_tool_message.content:
                return AIMessage(content="记忆保存操作已取消。")
            return AIMessage(content="记忆保存成功。")

        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "save_memory",
                    "args": {"content": "我喜欢Python"},
                    "id": "save-call-1",
                    "type": "tool_call",
                }
            ],
        )


class ReactGraphTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = TemporaryDirectory()
        self.database_path = (
            Path(self.temporary_directory.name) / "checkpoints.sqlite"
        )
        self.memory_database_path = (
            Path(self.temporary_directory.name) / "memories.sqlite"
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _open_memory_store(self):
        return open_sqlite_memory_store(
            self.memory_database_path,
            embeddings=DeterministicTestEmbeddings(),
            dimensions=TEST_EMBEDDING_DIMENSIONS,
        )

    def test_graph_preserves_react_loop_with_recovery_nodes(self):
        app = build_graph(DirectAnswerModel(), TOOLS)
        graph = app.get_graph()
        nodes = set(graph.nodes)
        edges = {(edge.source, edge.target) for edge in graph.edges}

        self.assertEqual(
            nodes,
            {
                "__start__",
                "memory_recall",
                "agent",
                "tools",
                "tool_success",
                "human_review",
                "failure",
                "memory_manager",
                "memory_executor",
                "__end__",
            },
        )
        self.assertTrue(
            {
                ("__start__", "memory_recall"),
                ("memory_recall", "agent"),
                ("agent", "tools"),
                ("agent", "memory_manager"),
                ("tools", "tool_success"),
                ("tool_success", "agent"),
                ("memory_manager", "memory_executor"),
                ("memory_executor", "__end__"),
            }.issubset(edges)
        )

    def test_graph_can_end_without_tool_call(self):
        app = build_graph(DirectAnswerModel(), TOOLS)

        result = app.invoke(
            {"messages": [HumanMessage(content="你好")]}
        )

        self.assertEqual(result["messages"][-1].content, "直接回答")

    def test_graph_executes_tool_and_returns_to_agent(self):
        llm = ToolCallingModel()
        with open_sqlite_checkpointer(self.database_path) as checkpointer:
            app = build_graph(llm, TOOLS, checkpointer=checkpointer)

            result = app.invoke(
                {"messages": [HumanMessage(content="计算 6 * 7")]},
                config={"configurable": {"thread_id": "user_001"}},
            )

            self.assertEqual(llm.invocation_count, 2)
            self.assertTrue(
                any(
                    isinstance(message, ToolMessage)
                    for message in result["messages"]
                )
            )
            self.assertEqual(result["messages"][-1].content, "计算结果是 42")

            history = list(
                app.get_state_history(
                    {"configurable": {"thread_id": "user_001"}}
                )
            )
            self.assertTrue(
                any(
                    snapshot.next == ("tools",)
                    and isinstance(snapshot.values["messages"][-1], AIMessage)
                    and snapshot.values["messages"][-1].tool_calls
                    for snapshot in history
                )
            )
            self.assertTrue(
                any(
                    snapshot.next == ("agent",)
                    and isinstance(snapshot.values["messages"][-1], ToolMessage)
                    for snapshot in history
                )
            )
            self.assertTrue(
                any(
                    snapshot.next == ()
                    and snapshot.values.get("messages")
                    and snapshot.values["messages"][-1].content
                    == "计算结果是 42"
                    for snapshot in history
                )
            )

    def test_same_thread_restores_messages_and_history_after_new_connection(self):
        config = {"configurable": {"thread_id": "user_001"}}

        with open_sqlite_checkpointer(self.database_path) as first_checkpointer:
            first_app = build_graph(
                NameMemoryModel(), TOOLS, checkpointer=first_checkpointer
            )
            first_app.invoke(
                {"messages": [HumanMessage(content="我的名字是小明")]},
                config=config,
            )
            history_before_restart = list(first_app.get_state_history(config))

        with open_sqlite_checkpointer(self.database_path) as second_checkpointer:
            second_app = build_graph(
                NameMemoryModel(), TOOLS, checkpointer=second_checkpointer
            )
            state_after_restart = second_app.get_state(config)
            history_after_restart = list(second_app.get_state_history(config))
            result = second_app.invoke(
                {"messages": [HumanMessage(content="我叫什么名字？")]},
                config=config,
            )

            self.assertTrue(
                any(
                    isinstance(message, HumanMessage)
                    and message.content == "我的名字是小明"
                    for message in state_after_restart.values["messages"]
                )
            )
            self.assertEqual(
                len(history_after_restart), len(history_before_restart)
            )
            self.assertEqual(
                result["messages"][-1].content, "你的名字是小明。"
            )
            human_messages = [
                message
                for message in result["messages"]
                if isinstance(message, HumanMessage)
            ]
            self.assertEqual(len(human_messages), 2)

    def test_different_threads_are_isolated(self):
        user_001 = {"configurable": {"thread_id": "user_001"}}
        user_002 = {"configurable": {"thread_id": "user_002"}}

        with open_sqlite_checkpointer(self.database_path) as checkpointer:
            app = build_graph(
                NameMemoryModel(), TOOLS, checkpointer=checkpointer
            )
            app.invoke(
                {"messages": [HumanMessage(content="我的名字是小明")]},
                config=user_001,
            )
            result = app.invoke(
                {"messages": [HumanMessage(content="我叫什么名字？")]},
                config=user_002,
            )

            self.assertEqual(
                result["messages"][-1].content, "我不知道你的名字。"
            )
            self.assertFalse(
                any(
                    isinstance(message, HumanMessage)
                    and "我的名字是小明" in message.content
                    for message in result["messages"]
                )
            )

    def test_sensitive_tool_pauses_and_executes_once_after_approval(self):
        config = {"configurable": {"thread_id": "thread_hitl_001"}}

        with (
            open_sqlite_checkpointer(self.database_path) as checkpointer,
            self._open_memory_store() as store,
        ):
            app = build_graph(
                SaveMemoryToolCallingModel(),
                TOOLS,
                checkpointer=checkpointer,
                store=store,
            )
            context = AgentContext(user_id="user_001")
            interrupted = app.invoke(
                {"messages": [HumanMessage(content="请记住，我喜欢Python")]},
                config=config,
                context=context,
            )

            self.assertEqual(
                list_memories_skill(store, user_id="user_001"), []
            )
            interrupt_info = interrupted["__interrupt__"][0]
            self.assertEqual(interrupt_info.value["action"], "save_user_memory")
            self.assertEqual(interrupt_info.value["tool_name"], "save_memory")
            self.assertEqual(
                interrupt_info.value["arguments"],
                {"content": "我喜欢Python", "memory_type": "fact"},
            )

            result = app.invoke(
                Command(resume={"approved": True}),
                config=config,
                context=context,
            )

            memories = list_memories_skill(store, user_id="user_001")
            self.assertEqual([memory["content"] for memory in memories], ["我喜欢Python"])
            self.assertEqual(result["messages"][-1].content, "记忆保存成功。")

    def test_sensitive_tool_rejection_has_no_side_effect(self):
        config = {"configurable": {"thread_id": "thread_hitl_002"}}

        with (
            open_sqlite_checkpointer(self.database_path) as checkpointer,
            self._open_memory_store() as store,
        ):
            app = build_graph(
                SaveMemoryToolCallingModel(),
                TOOLS,
                checkpointer=checkpointer,
                store=store,
            )
            context = AgentContext(user_id="user_001")
            interrupted = app.invoke(
                {"messages": [HumanMessage(content="请记住，我喜欢Python")]},
                config=config,
                context=context,
            )
            self.assertTrue(interrupted["__interrupt__"])

            result = app.invoke(
                Command(
                    resume={"approved": False, "reason": "用户拒绝"}
                ),
                config=config,
                context=context,
            )

            self.assertEqual(
                list_memories_skill(store, user_id="user_001"), []
            )
            self.assertEqual(
                result["messages"][-1].content, "记忆保存操作已取消。"
            )

    def test_pending_interrupt_survives_connection_restart_and_is_isolated(self):
        interrupted_config = {
            "configurable": {"thread_id": "thread_hitl_003"}
        }
        isolated_config = {
            "configurable": {"thread_id": "thread_hitl_004"}
        }

        with (
            open_sqlite_checkpointer(self.database_path) as first_checkpointer,
            self._open_memory_store() as first_store,
        ):
            first_app = build_graph(
                SaveMemoryToolCallingModel(),
                TOOLS,
                checkpointer=first_checkpointer,
                store=first_store,
            )
            context = AgentContext(user_id="user_001")
            first_app.invoke(
                {"messages": [HumanMessage(content="请记住，我喜欢Python")]},
                config=interrupted_config,
                context=context,
            )
            self.assertEqual(
                list_memories_skill(first_store, user_id="user_001"), []
            )

        with (
            open_sqlite_checkpointer(self.database_path) as second_checkpointer,
            self._open_memory_store() as second_store,
        ):
            second_app = build_graph(
                SaveMemoryToolCallingModel(),
                TOOLS,
                checkpointer=second_checkpointer,
                store=second_store,
            )
            pending_state = second_app.get_state(interrupted_config)
            isolated_state = second_app.get_state(isolated_config)

            self.assertTrue(pending_state.interrupts)
            self.assertEqual(pending_state.next, ("tools",))
            self.assertFalse(isolated_state.interrupts)
            self.assertEqual(isolated_state.values, {})

            result = second_app.invoke(
                Command(resume={"approved": True}),
                config=interrupted_config,
                context=context,
            )

            memories = list_memories_skill(second_store, user_id="user_001")
            self.assertEqual([memory["content"] for memory in memories], ["我喜欢Python"])
            self.assertEqual(result["messages"][-1].content, "记忆保存成功。")


if __name__ == "__main__":
    unittest.main()
