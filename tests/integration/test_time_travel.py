import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from ai_agent_learning.agent import (
    ThreadIsolationError,
    UnsafeTimeTravelError,
    build_graph,
    checkpoint_id,
    fork_calculation_result,
    replay_checkpoint,
    select_checkpoint,
    show_state_history,
)
from ai_agent_learning.checkpoint import open_sqlite_checkpointer
from ai_agent_learning.skills import calculate as calculate_skill
from ai_agent_learning.skills.memory import (
    clear_saved_memories,
    get_saved_memories,
)
from ai_agent_learning.tools import TOOLS


class StateDrivenCalculateModel:
    """Deterministic model whose decision depends only on persisted messages."""

    def __init__(self):
        self.invocation_count = 0

    def bind_tools(self, _tools):
        return self

    def invoke(self, messages):
        self.invocation_count += 1
        last_message = messages[-1]
        if isinstance(last_message, ToolMessage):
            return AIMessage(content=f"计算结果是 {last_message.content}")

        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "calculate",
                    "args": {"expression": "6 * 7"},
                    "id": "calculate-call-1",
                    "type": "tool_call",
                }
            ],
        )


class StateDrivenSaveMemoryModel:
    def bind_tools(self, _tools):
        return self

    def invoke(self, messages):
        if isinstance(messages[-1], ToolMessage):
            return AIMessage(content="记忆操作已处理。")

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


class TimeTravelTests(unittest.TestCase):
    def setUp(self):
        clear_saved_memories()
        self.temporary_directory = TemporaryDirectory()
        self.database_path = (
            Path(self.temporary_directory.name) / "checkpoints.sqlite"
        )

    def tearDown(self):
        clear_saved_memories()
        self.temporary_directory.cleanup()

    @staticmethod
    def _config(thread_id: str) -> dict:
        return {"configurable": {"thread_id": thread_id}}

    @staticmethod
    def _find_snapshot(history, *, next_nodes, last_message_type):
        return next(
            snapshot
            for snapshot in history
            if snapshot.next == next_nodes
            and snapshot.values.get("messages")
            and isinstance(snapshot.values["messages"][-1], last_message_type)
        )

    def test_history_and_replay_execute_only_nodes_after_checkpoint(self):
        thread_id = "time_travel_replay"
        config = self._config(thread_id)
        model = StateDrivenCalculateModel()

        with patch(
            "ai_agent_learning.tools.adapters.calculate_skill",
            wraps=calculate_skill,
        ) as calculate_mock:
            with open_sqlite_checkpointer(self.database_path) as checkpointer:
                app = build_graph(model, TOOLS, checkpointer=checkpointer)
                app.invoke(
                    {"messages": [HumanMessage(content="计算 6 * 7")]},
                    config=config,
                )
                history = list(app.get_state_history(config))

                output = io.StringIO()
                with redirect_stdout(output):
                    displayed_history = show_state_history(app, thread_id)
                rendered = output.getvalue()

                self.assertEqual(displayed_history, history)
                self.assertIn("checkpoint_id:", rendered)
                self.assertIn("metadata.step:", rendered)
                self.assertIn("metadata.source:", rendered)
                self.assertIn("下一步执行节点:", rendered)

                before_tools = self._find_snapshot(
                    history,
                    next_nodes=("tools",),
                    last_message_type=AIMessage,
                )
                model_calls_before = model.invocation_count
                tool_calls_before = calculate_mock.call_count

                replay_result = replay_checkpoint(app, before_tools, thread_id)

                self.assertEqual(
                    model.invocation_count,
                    model_calls_before + 1,
                    "产生历史 tool_call 的 AgentNode 不应重新执行",
                )
                self.assertEqual(
                    calculate_mock.call_count,
                    tool_calls_before + 1,
                    "Checkpoint 的 next=tools，因此 ToolNode 应重新执行",
                )
                self.assertEqual(
                    replay_result["messages"][-1].content,
                    "计算结果是 42",
                )
                replay_history = list(app.get_state_history(config))
                self.assertTrue(
                    any(
                        (snapshot.metadata or {}).get("source") == "fork"
                        for snapshot in replay_history
                    )
                )

                before_agent = self._find_snapshot(
                    history,
                    next_nodes=("agent",),
                    last_message_type=ToolMessage,
                )
                tool_calls_before = calculate_mock.call_count
                replay_checkpoint(app, before_agent, thread_id)
                self.assertEqual(
                    calculate_mock.call_count,
                    tool_calls_before,
                    "Checkpoint 的 next=agent，因此之前的 ToolNode 不应重新执行",
                )

    def test_fork_replaces_one_tool_message_without_overwriting_original(self):
        thread_id = "time_travel_fork"
        config = self._config(thread_id)
        model = StateDrivenCalculateModel()

        with open_sqlite_checkpointer(self.database_path) as checkpointer:
            app = build_graph(model, TOOLS, checkpointer=checkpointer)
            app.invoke(
                {"messages": [HumanMessage(content="计算 6 * 7")]},
                config=config,
            )
            history = list(app.get_state_history(config))
            selected = self._find_snapshot(
                history,
                next_nodes=("agent",),
                last_message_type=ToolMessage,
            )
            original_checkpoint_id = checkpoint_id(selected)
            original_message = selected.values["messages"][-1]

            fork_config, result = fork_calculation_result(
                app,
                selected,
                thread_id,
                "43",
            )

            self.assertNotEqual(
                fork_config["configurable"]["checkpoint_id"],
                original_checkpoint_id,
            )
            self.assertEqual(result["messages"][-1].content, "计算结果是 43")

            update_snapshot = app.get_state(fork_config)
            updated_messages = update_snapshot.values["messages"]
            self.assertEqual(len(updated_messages), len(selected.values["messages"]))
            self.assertEqual(updated_messages[-1].content, "43")
            self.assertEqual(updated_messages[-1].id, original_message.id)
            self.assertEqual((update_snapshot.metadata or {}).get("source"), "update")
            self.assertEqual(update_snapshot.next, ("tool_success",))

            original_snapshot = app.get_state(selected.config)
            self.assertEqual(original_snapshot.values["messages"][-1].content, "42")
            self.assertEqual(checkpoint_id(original_snapshot), original_checkpoint_id)

    def test_original_and_fork_history_survive_sqlite_restart(self):
        thread_id = "time_travel_restart"
        config = self._config(thread_id)
        original_checkpoint_id = ""
        fork_checkpoint_id = ""

        with open_sqlite_checkpointer(self.database_path) as checkpointer:
            app = build_graph(
                StateDrivenCalculateModel(), TOOLS, checkpointer=checkpointer
            )
            app.invoke(
                {"messages": [HumanMessage(content="计算 6 * 7")]},
                config=config,
            )
            selected = self._find_snapshot(
                list(app.get_state_history(config)),
                next_nodes=("agent",),
                last_message_type=ToolMessage,
            )
            original_checkpoint_id = checkpoint_id(selected)
            fork_config, _ = fork_calculation_result(
                app, selected, thread_id, "43"
            )
            fork_checkpoint_id = fork_config["configurable"]["checkpoint_id"]

        with open_sqlite_checkpointer(self.database_path) as checkpointer:
            restarted_app = build_graph(
                StateDrivenCalculateModel(), TOOLS, checkpointer=checkpointer
            )
            persisted_ids = {
                checkpoint_id(snapshot)
                for snapshot in restarted_app.get_state_history(config)
            }

            self.assertIn(original_checkpoint_id, persisted_ids)
            self.assertIn(fork_checkpoint_id, persisted_ids)

    def test_checkpoint_cannot_be_replayed_under_another_thread(self):
        owner_thread = "time_travel_owner"
        other_thread = "time_travel_other"
        owner_config = self._config(owner_thread)

        with open_sqlite_checkpointer(self.database_path) as checkpointer:
            app = build_graph(
                StateDrivenCalculateModel(), TOOLS, checkpointer=checkpointer
            )
            app.invoke(
                {"messages": [HumanMessage(content="计算 6 * 7")]},
                config=owner_config,
            )
            selected = self._find_snapshot(
                list(app.get_state_history(owner_config)),
                next_nodes=("tools",),
                last_message_type=AIMessage,
            )

            with self.assertRaises(ThreadIsolationError):
                replay_checkpoint(app, selected, other_thread)

            self.assertEqual(
                list(app.get_state_history(self._config(other_thread))),
                [],
            )

    def test_sensitive_checkpoint_is_rejected_without_side_effect(self):
        thread_id = "time_travel_sensitive"
        config = self._config(thread_id)

        with open_sqlite_checkpointer(self.database_path) as checkpointer:
            app = build_graph(
                StateDrivenSaveMemoryModel(), TOOLS, checkpointer=checkpointer
            )
            interrupted = app.invoke(
                {"messages": [HumanMessage(content="请记住，我喜欢Python")]},
                config=config,
            )
            self.assertTrue(interrupted["__interrupt__"])
            selected = next(
                snapshot
                for snapshot in app.get_state_history(config)
                if snapshot.next == ("tools",)
            )

            with self.assertRaises(UnsafeTimeTravelError):
                replay_checkpoint(app, selected, thread_id)

            self.assertEqual(get_saved_memories(), ())

    def test_checkpoint_selection_uses_one_based_display_sequence(self):
        snapshots = [object(), object(), object()]
        self.assertIs(select_checkpoint(snapshots, 2), snapshots[1])


if __name__ == "__main__":
    unittest.main()
