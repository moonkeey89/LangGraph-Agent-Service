import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.runtime import Runtime
from langgraph.types import Command

from ai_agent_learning.agent import AgentContext, MemoryDecision, build_graph
from ai_agent_learning.agent.memory_manager import MemoryExecutorNode
from ai_agent_learning.checkpoint import open_sqlite_checkpointer
from ai_agent_learning.memory_store import open_sqlite_memory_store
from ai_agent_learning.skills.memory import (
    get_memory,
    list_memories,
    save_memory,
    search_memory,
)
from ai_agent_learning.tools import TOOLS
from tests.helpers import DeterministicTestEmbeddings, TEST_EMBEDDING_DIMENSIONS


class FinalAnswerModel:
    def bind_tools(self, _tools):
        return self

    def invoke(self, _messages):
        return AIMessage(content="主Agent回答完成")


class ExplicitSaveModel:
    def bind_tools(self, _tools):
        return self

    def invoke(self, messages):
        if isinstance(messages[-1], ToolMessage):
            return AIMessage(content="显式记忆保存完成")
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "save_memory",
                    "args": {
                        "content": "模型参数不可信",
                        "memory_type": "preference",
                    },
                    "id": "explicit-memory-1",
                    "type": "tool_call",
                }
            ],
        )


class OvereagerSaveThenRecallModel:
    """Reproduces a real LLM calling save_memory without explicit intent."""

    def bind_tools(self, _tools):
        return self

    def invoke(self, messages):
        last_message = messages[-1]
        if isinstance(last_message, ToolMessage):
            return AIMessage(content="已收到记忆工具结果")

        human_message = next(
            message
            for message in reversed(messages)
            if isinstance(message, HumanMessage)
        )
        if "我爱吃青菜" in str(human_message.content):
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "save_memory",
                        "args": {
                            "content": "用户爱吃青菜并且最喜欢数学",
                            "memory_type": "preference",
                        },
                        "id": "overeager-save-1",
                        "type": "tool_call",
                    }
                ],
            )

        recalled_text = "\n".join(
            str(message.content)
            for message in messages
            if isinstance(message, SystemMessage)
        )
        if "青菜" in recalled_text:
            return AIMessage(content="你喜欢吃青菜，而且最喜欢数学。")
        return AIMessage(content="我不知道你喜欢什么。")


class ScriptedDecisionModel:
    def __init__(self, decision=None, error: Exception | None = None):
        self.decision = decision
        self.error = error
        self.invocation_count = 0
        self.inputs = []
        self.schema = None

    def with_structured_output(self, schema, **_kwargs):
        self.schema = schema
        return self

    def invoke(self, messages):
        self.invocation_count += 1
        self.inputs.append(messages)
        if self.error is not None:
            raise self.error
        return self.decision


def decision(
    operation: str,
    *,
    content: str = "",
    target_memory_id: str | None = None,
    memory_type: str = "fact",
    confidence: float = 0.95,
    reason: str = "测试决定",
) -> MemoryDecision:
    return MemoryDecision(
        operation=operation,
        memory_type=memory_type,
        content=content,
        target_memory_id=target_memory_id,
        confidence=confidence,
        reason=reason,
    )


class MemoryManagerIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = TemporaryDirectory()
        root = Path(self.temporary_directory.name)
        self.checkpoint_path = root / "checkpoints.sqlite"
        self.memory_path = root / "memories.sqlite"

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _store(self):
        return open_sqlite_memory_store(
            self.memory_path,
            embeddings=DeterministicTestEmbeddings(),
            dimensions=TEST_EMBEDDING_DIMENSIONS,
        )

    @staticmethod
    def _config(thread_id: str) -> dict:
        return {"configurable": {"thread_id": thread_id}}

    @staticmethod
    def _context(user_id: str = "user_001") -> AgentContext:
        return AgentContext(user_id=user_id)

    def _invoke(self, app, text: str, thread_id: str, user_id: str = "user_001"):
        return app.invoke(
            {"messages": [HumanMessage(content=text)]},
            config=self._config(thread_id),
            context=self._context(user_id),
        )

    def test_add_duplicate_and_lightweight_none(self):
        manager_model = ScriptedDecisionModel(
            decision(
                "ADD",
                content="用户平时使用PyCharm开发Python项目",
                memory_type="preference",
            )
        )
        with (
            open_sqlite_checkpointer(self.checkpoint_path) as checkpointer,
            self._store() as store,
        ):
            app = build_graph(
                FinalAnswerModel(),
                TOOLS,
                checkpointer=checkpointer,
                store=store,
                memory_manager_llm=manager_model,
            )
            added = self._invoke(
                app,
                "我平时使用PyCharm开发Python项目",
                "thread_A",
            )
            duplicate = self._invoke(
                app,
                "我平时使用PyCharm开发Python项目",
                "thread_B",
            )
            calls_before_calculation = manager_model.invocation_count
            calculation = self._invoke(
                app,
                "帮我计算100×20",
                "thread_calculate",
            )

            memories = list_memories(store, user_id="user_001")
            self.assertEqual(len(memories), 1)
            self.assertEqual(memories[0]["source"], "memory_manager")
            self.assertEqual(memories[0]["source_thread_id"], "thread_A")
            self.assertEqual(added["memory_manager_status"], "applied")
            self.assertEqual(
                duplicate["memory_decision"]["operation"], "NONE"
            )
            self.assertEqual(
                calculation["memory_decision"]["operation"], "NONE"
            )
            self.assertEqual(
                manager_model.invocation_count, calls_before_calculation
            )

    def test_update_then_delete_replaces_one_user_scoped_memory(self):
        manager_model = ScriptedDecisionModel()
        with (
            open_sqlite_checkpointer(self.checkpoint_path) as checkpointer,
            self._store() as store,
        ):
            save_memory(
                store,
                user_id="user_001",
                memory_id="editor-memory",
                content="用户使用PyCharm开发",
                memory_type="preference",
                source_thread_id="seed",
            )
            app = build_graph(
                FinalAnswerModel(),
                TOOLS,
                checkpointer=checkpointer,
                store=store,
                memory_manager_llm=manager_model,
            )

            manager_model.decision = decision(
                "UPDATE",
                content="用户现在使用VS Code开发",
                target_memory_id="editor-memory",
                memory_type="preference",
            )
            updated_result = self._invoke(
                app,
                "我现在改用VS Code开发了",
                "thread_update",
            )
            memories = list_memories(store, user_id="user_001")
            self.assertEqual(len(memories), 1)
            self.assertEqual(memories[0]["content"], "用户现在使用VS Code开发")
            self.assertEqual(updated_result["memory_manager_status"], "applied")

            manager_model.decision = decision(
                "DELETE",
                target_memory_id="editor-memory",
                memory_type="preference",
            )
            deleted_result = self._invoke(
                app,
                "忘记我使用VS Code开发这件事",
                "thread_delete",
            )
            self.assertEqual(deleted_result["memory_manager_status"], "applied")
            self.assertEqual(list_memories(store, user_id="user_001"), [])

    def test_policy_rejects_sensitive_content_and_cross_user_target(self):
        with self._store() as store:
            sensitive_state = {
                "messages": [
                    HumanMessage(content="我使用的API Key是sk-test-123456789"),
                    AIMessage(content="主Agent回答完成"),
                ],
                "status": "completed",
                "memory_decision": decision(
                    "ADD",
                    content="用户API Key是sk-test-123456789",
                ).model_dump(),
                "memory_candidate_ids": [],
            }
            rejected = MemoryExecutorNode().run(
                sensitive_state,
                Runtime(context=self._context(), store=store),
            )
            self.assertEqual(rejected["memory_manager_status"], "rejected")
            self.assertEqual(rejected["memory_decision"]["operation"], "NONE")
            self.assertEqual(list_memories(store, user_id="user_001"), [])

            low_confidence_state = {
                **sensitive_state,
                "messages": [
                    HumanMessage(content="我喜欢Python"),
                    AIMessage(content="主Agent回答完成"),
                ],
                "memory_decision": decision(
                    "ADD",
                    content="用户喜欢Python",
                    confidence=0.5,
                ).model_dump(),
            }
            low_confidence = MemoryExecutorNode().run(
                low_confidence_state,
                Runtime(context=self._context(), store=store),
            )
            self.assertEqual(
                low_confidence["memory_manager_status"], "rejected"
            )
            self.assertEqual(list_memories(store, user_id="user_001"), [])

            save_memory(
                store,
                user_id="user_001",
                memory_id="private-memory",
                content="用户使用Python",
                memory_type="preference",
                source_thread_id="thread_A",
            )

        malicious_manager = ScriptedDecisionModel(
            decision(
                "DELETE",
                target_memory_id="private-memory",
                memory_type="preference",
            )
        )
        with (
            open_sqlite_checkpointer(self.checkpoint_path) as checkpointer,
            self._store() as store,
        ):
            app = build_graph(
                FinalAnswerModel(),
                TOOLS,
                checkpointer=checkpointer,
                store=store,
                memory_manager_llm=malicious_manager,
            )
            result = self._invoke(
                app,
                "忘记我使用Python这件事",
                "thread_other_user",
                user_id="user_002",
            )
            self.assertEqual(result["memory_manager_status"], "rejected")
            self.assertIsNotNone(
                get_memory(
                    store,
                    user_id="user_001",
                    memory_id="private-memory",
                )
            )
            payload = json.loads(
                str(malicious_manager.inputs[0][-1].content)
            )
            self.assertEqual(payload["candidate_memories"], [])

    def test_cross_thread_and_restart_preserve_managed_memory(self):
        manager_model = ScriptedDecisionModel(
            decision(
                "ADD",
                content="用户的目标是学习LangGraph",
                memory_type="profile",
            )
        )
        with (
            open_sqlite_checkpointer(self.checkpoint_path) as checkpointer,
            self._store() as store,
        ):
            app = build_graph(
                FinalAnswerModel(),
                TOOLS,
                checkpointer=checkpointer,
                store=store,
                memory_manager_llm=manager_model,
            )
            self._invoke(app, "我的目标是学习LangGraph", "thread_A")

        with self._store() as restarted_store:
            memories = search_memory(
                restarted_store,
                user_id="user_001",
                query="学习目标",
            )
            self.assertEqual(len(memories), 1)
            self.assertIn("LangGraph", memories[0]["content"])
            self.assertEqual(
                search_memory(
                    restarted_store,
                    user_id="user_002",
                    query="学习目标",
                ),
                [],
            )

    def test_explicit_save_is_not_duplicated_by_manager(self):
        manager_model = ScriptedDecisionModel(
            decision(
                "ADD",
                content="我喜欢Python",
                memory_type="preference",
            )
        )
        config = self._config("thread_explicit")
        context = self._context()
        with (
            open_sqlite_checkpointer(self.checkpoint_path) as checkpointer,
            self._store() as store,
        ):
            app = build_graph(
                ExplicitSaveModel(),
                TOOLS,
                checkpointer=checkpointer,
                store=store,
                memory_manager_llm=manager_model,
            )
            interrupted = app.invoke(
                {"messages": [HumanMessage(content="请记住，我喜欢Python")]},
                config=config,
                context=context,
            )
            self.assertTrue(interrupted["__interrupt__"])
            result = app.invoke(
                Command(resume={"approved": True}),
                config=config,
                context=context,
            )

            self.assertEqual(len(list_memories(store, user_id="user_001")), 1)
            self.assertEqual(manager_model.invocation_count, 0)
            self.assertEqual(result["memory_decision"]["operation"], "NONE")

    def test_manager_failure_degrades_to_none_without_losing_answer(self):
        manager_model = ScriptedDecisionModel(
            error=ValueError("invalid structured output")
        )
        with (
            open_sqlite_checkpointer(self.checkpoint_path) as checkpointer,
            self._store() as store,
        ):
            app = build_graph(
                FinalAnswerModel(),
                TOOLS,
                checkpointer=checkpointer,
                store=store,
                memory_manager_llm=manager_model,
            )
            result = self._invoke(
                app,
                "我喜欢使用类型检查工具",
                "thread_failure",
            )

            self.assertEqual(result["messages"][-1].content, "主Agent回答完成")
            self.assertEqual(result["memory_manager_status"], "failed")
            self.assertEqual(result["memory_decision"]["operation"], "NONE")
            self.assertIn(
                "invalid structured output", result["memory_manager_error"]
            )
            self.assertEqual(list_memories(store, user_id="user_001"), [])

    def test_non_explicit_tool_refusal_falls_through_and_recalls_cross_thread(self):
        manager_model = ScriptedDecisionModel(
            decision(
                "ADD",
                content="用户爱吃青菜并且最喜欢数学",
                memory_type="preference",
            )
        )
        context = self._context("user_001")

        with (
            open_sqlite_checkpointer(self.checkpoint_path) as checkpointer,
            self._store() as store,
        ):
            first_app = build_graph(
                OvereagerSaveThenRecallModel(),
                TOOLS,
                checkpointer=checkpointer,
                store=store,
                memory_manager_llm=manager_model,
            )
            first_result = self._invoke(
                first_app,
                "我爱吃青菜，最喜欢数学",
                "thread_001",
            )

            tool_messages = [
                message
                for message in first_result["messages"]
                if isinstance(message, ToolMessage)
            ]
            self.assertTrue(
                any("未检测到" in str(message.content) for message in tool_messages)
            )
            self.assertEqual(first_result["memory_manager_status"], "applied")
            self.assertEqual(len(list_memories(store, user_id="user_001")), 1)

        # Reopen both SQLite databases and use a new thread for the same user.
        with (
            open_sqlite_checkpointer(self.checkpoint_path) as checkpointer,
            self._store() as store,
        ):
            restarted_app = build_graph(
                OvereagerSaveThenRecallModel(),
                TOOLS,
                checkpointer=checkpointer,
                store=store,
                memory_manager_llm=manager_model,
            )
            recalled = self._invoke(
                restarted_app,
                "我喜欢什么",
                "thread_002",
            )
            isolated = self._invoke(
                restarted_app,
                "我喜欢什么",
                "thread_other_user",
                user_id="user_002",
            )

            self.assertEqual(
                recalled["messages"][-1].content,
                "你喜欢吃青菜，而且最喜欢数学。",
            )
            self.assertEqual(recalled["memory_recall_status"], "completed")
            self.assertEqual(len(recalled["recalled_memories"]), 1)
            self.assertEqual(isolated["messages"][-1].content, "我不知道你喜欢什么。")
            self.assertEqual(isolated["recalled_memories"], [])


if __name__ == "__main__":
    unittest.main()
