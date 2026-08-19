import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import Mock

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import StateSnapshot

from ai_agent_learning.agent.checkpoint_debug import (
    show_current_state,
    show_state_history,
)


def make_snapshot(
    messages,
    *,
    checkpoint_id: str,
    next_nodes: tuple[str, ...],
    created_at: str,
    parent_config=None,
):
    return StateSnapshot(
        values={"messages": messages},
        next=next_nodes,
        config={
            "configurable": {
                "thread_id": "user_001",
                "checkpoint_id": checkpoint_id,
            }
        },
        metadata={"source": "loop", "step": 1},
        created_at=created_at,
        parent_config=parent_config,
        tasks=(),
        interrupts=(),
    )


class CheckpointDebugTests(unittest.TestCase):
    def test_show_current_state_outputs_requested_snapshot_fields(self):
        graph = Mock()
        snapshot = make_snapshot(
            [HumanMessage(content="你好"), AIMessage(content="你好！")],
            checkpoint_id="checkpoint-002",
            next_nodes=(),
            created_at="2026-08-18T10:00:00+00:00",
            parent_config={"configurable": {"checkpoint_id": "checkpoint-001"}},
        )
        graph.get_state.return_value = snapshot
        output = io.StringIO()

        with redirect_stdout(output):
            result = show_current_state(graph, "user_001")

        self.assertIs(result, snapshot)
        graph.get_state.assert_called_once_with(
            {"configurable": {"thread_id": "user_001"}}
        )
        rendered = output.getvalue()
        for field in (
            "snapshot.values",
            "snapshot.next",
            "snapshot.config",
            "snapshot.metadata",
            "snapshot.created_at",
            "snapshot.parent_config",
            "snapshot.interrupts",
        ):
            self.assertIn(field, rendered)

    def test_show_state_history_outputs_checkpoint_summary(self):
        graph = Mock()
        newest = make_snapshot(
            [HumanMessage(content="计算 6 * 7"), AIMessage(content="计算结果是 42")],
            checkpoint_id="checkpoint-002",
            next_nodes=(),
            created_at="2026-08-18T10:00:02+00:00",
        )
        older = make_snapshot(
            [HumanMessage(content="计算 6 * 7")],
            checkpoint_id="checkpoint-001",
            next_nodes=("agent",),
            created_at="2026-08-18T10:00:01+00:00",
        )
        graph.get_state_history.return_value = iter([newest, older])
        output = io.StringIO()

        with redirect_stdout(output):
            result = show_state_history(graph, "user_001")

        self.assertEqual(result, [newest, older])
        graph.get_state_history.assert_called_once_with(
            {"configurable": {"thread_id": "user_001"}}
        )
        rendered = output.getvalue()
        self.assertIn("Checkpoint #1（最新）", rendered)
        self.assertIn("Checkpoint #2", rendered)
        self.assertIn("消息数量: 2", rendered)
        self.assertIn("最后一条消息类型: AIMessage", rendered)
        self.assertIn("最后一条消息内容: 计算结果是 42", rendered)
        self.assertIn("下一步执行节点: agent", rendered)
        self.assertIn("checkpoint_id: checkpoint-002", rendered)


if __name__ == "__main__":
    unittest.main()
