import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.types import Command

from ai_agent_learning.agent import AgentContext, PermanentToolError, build_graph
from ai_agent_learning.checkpoint import open_sqlite_checkpointer
from ai_agent_learning.memory_store import open_sqlite_memory_store
from ai_agent_learning.skills.unstable import (
    get_unstable_attempts,
    reset_unstable_tool,
    set_unstable_always_timeout,
)
from ai_agent_learning.tools import TOOLS
from tests.helpers import DeterministicTestEmbeddings, TEST_EMBEDDING_DIMENSIONS


@tool
def requires_integer(value: int) -> str:
    """Return an integer after schema validation."""
    return f"有效参数：{value}"


@tool
def permission_tool(resource: str) -> str:
    """Always fail with a permission error."""
    raise PermissionError(f"无权访问：{resource}")


@tool
def permanent_tool(resource: str) -> str:
    """Always fail permanently."""
    raise PermanentToolError(f"资源永久不存在：{resource}")


class UnstableToolModel:
    def __init__(self, task: str = "demo"):
        self.task = task
        self.invocation_count = 0

    def bind_tools(self, _tools):
        return self

    def invoke(self, messages):
        self.invocation_count += 1
        last_message = messages[-1]
        if isinstance(last_message, ToolMessage):
            return AIMessage(content=f"最终结果：{last_message.content}")
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "unstable_tool",
                    "args": {"task": self.task},
                    "id": "unstable-call-1",
                    "type": "tool_call",
                }
            ],
        )


class CorrectingArgumentsModel:
    def __init__(self):
        self.invocation_count = 0

    def bind_tools(self, _tools):
        return self

    def invoke(self, messages):
        self.invocation_count += 1
        last_message = messages[-1]
        if isinstance(last_message, ToolMessage):
            if last_message.status == "error":
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "requires_integer",
                            "args": {"value": 7},
                            "id": "corrected-call-2",
                            "type": "tool_call",
                        }
                    ],
                )
            return AIMessage(content=f"修正成功：{last_message.content}")

        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "requires_integer",
                    "args": {"value": "不是整数"},
                    "id": "invalid-call-1",
                    "type": "tool_call",
                }
            ],
        )


class FailingToolModel:
    def __init__(self, tool_name: str):
        self.tool_name = tool_name
        self.invocation_count = 0

    def bind_tools(self, _tools):
        return self

    def invoke(self, _messages):
        self.invocation_count += 1
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": self.tool_name,
                    "args": {"resource": "secret"},
                    "id": f"{self.tool_name}-call-1",
                    "type": "tool_call",
                }
            ],
        )


class SaveMemoryModel:
    def bind_tools(self, _tools):
        return self

    def invoke(self, _messages):
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "save_memory",
                    "args": {"content": "我喜欢Python"},
                    "id": "save-call-unknown",
                    "type": "tool_call",
                }
            ],
        )


class ErrorRecoveryTests(unittest.TestCase):
    def setUp(self):
        reset_unstable_tool()
        self.temporary_directory = TemporaryDirectory()
        self.database_path = (
            Path(self.temporary_directory.name) / "checkpoints.sqlite"
        )
        self.memory_database_path = (
            Path(self.temporary_directory.name) / "memories.sqlite"
        )

    def tearDown(self):
        reset_unstable_tool()
        self.temporary_directory.cleanup()

    @staticmethod
    def _config(thread_id: str) -> dict:
        return {"configurable": {"thread_id": thread_id}}

    def test_unstable_tool_fails_twice_then_succeeds_with_persisted_counts(self):
        task = "eventual-success"
        config = self._config("retry_success")
        model = UnstableToolModel(task)

        with open_sqlite_checkpointer(self.database_path) as checkpointer:
            app = build_graph(model, TOOLS, checkpointer=checkpointer)
            result = app.invoke(
                {"messages": [HumanMessage(content="运行不稳定工具")]},
                config=config,
            )
            history = list(app.get_state_history(config))

        self.assertEqual(get_unstable_attempts(task), 3)
        self.assertEqual(model.invocation_count, 2)
        self.assertIn("执行成功", result["messages"][-1].content)
        observed_counts = {
            snapshot.values.get("retry_count") for snapshot in history
        }
        self.assertTrue({0, 1, 2}.issubset(observed_counts))
        self.assertEqual(result["retry_count"], 0)
        self.assertIsNone(result["error"])
        self.assertIsNone(result["error_type"])

    def test_retry_count_survives_sqlite_connection_restart(self):
        task = "restart-after-first-failure"
        config = self._config("retry_restart")

        with open_sqlite_checkpointer(self.database_path) as checkpointer:
            first_app = build_graph(
                UnstableToolModel(task), TOOLS, checkpointer=checkpointer
            )
            first_app.invoke(
                {"messages": [HumanMessage(content="运行不稳定工具")]},
                config=config,
                interrupt_after=["tools"],
            )
            self.assertEqual(first_app.get_state(config).values["retry_count"], 1)

        reset_unstable_tool()
        with open_sqlite_checkpointer(self.database_path) as checkpointer:
            restarted_app = build_graph(
                UnstableToolModel(task), TOOLS, checkpointer=checkpointer
            )
            restored = restarted_app.get_state(config)

            self.assertEqual(restored.values["retry_count"], 1)
            self.assertEqual(restored.values["error_type"], "transient")
            self.assertEqual(restored.next, ("tools",))

    def test_retry_limit_interrupts_and_cancel_does_not_execute_tool_again(self):
        task = "always-timeout"
        config = self._config("retry_cancel")
        set_unstable_always_timeout(True)

        with open_sqlite_checkpointer(self.database_path) as checkpointer:
            app = build_graph(
                UnstableToolModel(task), TOOLS, checkpointer=checkpointer
            )
            interrupted = app.invoke(
                {"messages": [HumanMessage(content="运行始终超时工具")]},
                config=config,
            )

            self.assertEqual(get_unstable_attempts(task), 3)
            interrupt_info = interrupted["__interrupt__"][0]
            self.assertEqual(interrupt_info.value["failed_node"], "tools")
            self.assertEqual(interrupt_info.value["retry_count"], 3)
            self.assertEqual(interrupt_info.value["options"], ["retry", "cancel"])

            result = app.invoke(
                Command(
                    resume={"action": "cancel", "reason": "用户取消重试"}
                ),
                config=config,
            )

            self.assertEqual(get_unstable_attempts(task), 3)
            self.assertEqual(result["status"], "cancelled")
            self.assertIn("未再进行重试", result["messages"][-1].content)

    def test_human_retry_continues_from_same_interrupted_tool(self):
        task = "manual-retry"
        config = self._config("retry_manual")
        set_unstable_always_timeout(True)

        with open_sqlite_checkpointer(self.database_path) as checkpointer:
            app = build_graph(
                UnstableToolModel(task), TOOLS, checkpointer=checkpointer
            )
            interrupted = app.invoke(
                {"messages": [HumanMessage(content="运行始终超时工具")]},
                config=config,
            )
            self.assertTrue(interrupted["__interrupt__"])

            set_unstable_always_timeout(False)
            result = app.invoke(
                Command(resume={"action": "retry"}),
                config=config,
            )

            self.assertEqual(get_unstable_attempts(task), 4)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["retry_count"], 0)
            self.assertIn("执行成功", result["messages"][-1].content)

    def test_invalid_arguments_return_to_agent_for_correction(self):
        config = self._config("invalid_arguments")
        model = CorrectingArgumentsModel()

        with open_sqlite_checkpointer(self.database_path) as checkpointer:
            app = build_graph(
                model,
                [*TOOLS, requires_integer],
                checkpointer=checkpointer,
            )
            result = app.invoke(
                {"messages": [HumanMessage(content="使用整数工具")]},
                config=config,
            )
            history = list(app.get_state_history(config))

        self.assertEqual(model.invocation_count, 3)
        self.assertIn("修正成功", result["messages"][-1].content)
        self.assertTrue(
            any(
                snapshot.values.get("status") == "agent_correction"
                and snapshot.values.get("error_type") == "invalid_arguments"
                and snapshot.values.get("retry_count") == 0
                for snapshot in history
            )
        )

    def test_permission_and_permanent_errors_are_not_retried(self):
        cases = [
            ("permission_tool", permission_tool, "permission"),
            ("permanent_tool", permanent_tool, "permanent"),
        ]

        for tool_name, failing_tool, expected_type in cases:
            with self.subTest(tool=tool_name):
                model = FailingToolModel(tool_name)
                config = self._config(f"non_retryable_{tool_name}")
                with open_sqlite_checkpointer(self.database_path) as checkpointer:
                    app = build_graph(
                        model,
                        [*TOOLS, failing_tool],
                        checkpointer=checkpointer,
                    )
                    result = app.invoke(
                        {"messages": [HumanMessage(content="执行失败工具")]},
                        config=config,
                    )

                self.assertEqual(model.invocation_count, 1)
                self.assertEqual(result["error_type"], expected_type)
                self.assertEqual(result["status"], "failed")
                self.assertIn("未进行自动重试", result["messages"][-1].content)

    def test_unknown_side_effect_result_is_not_retried(self):
        config = self._config("side_effect_unknown")
        side_effects: list[str] = []

        def write_then_timeout(_store, **kwargs) -> str:
            side_effects.append(kwargs["content"])
            raise TimeoutError("写入后连接中断，结果未知")

        with (
            patch(
                "ai_agent_learning.tools.adapters.save_memory_skill",
                side_effect=write_then_timeout,
            ),
            open_sqlite_checkpointer(self.database_path) as checkpointer,
            open_sqlite_memory_store(
                self.memory_database_path,
                embeddings=DeterministicTestEmbeddings(),
                dimensions=TEST_EMBEDDING_DIMENSIONS,
            ) as store,
        ):
            app = build_graph(
                SaveMemoryModel(),
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
                Command(resume={"approved": True}),
                config=config,
                context=context,
            )

        self.assertEqual(side_effects, ["我喜欢Python"])
        self.assertEqual(result["error_type"], "side_effect_unknown")
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["retry_count"], 0)


if __name__ == "__main__":
    unittest.main()
