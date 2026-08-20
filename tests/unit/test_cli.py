import unittest
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command, Interrupt

from ai_agent_learning.cli import (
    DEFAULT_THREAD_ID,
    DEFAULT_USER_ID,
    create_agent_app,
    prompt_thread_id,
    prompt_user_id,
    run_fork_command,
    run_cli,
    run_replay_command,
)


class CliTests(unittest.TestCase):
    @patch("ai_agent_learning.cli.build_graph")
    @patch("ai_agent_learning.cli.create_llm")
    def test_create_agent_app_injects_provided_checkpointer(
        self,
        create_llm,
        build_graph,
    ):
        settings = Mock()
        llm = create_llm.return_value
        checkpointer = Mock()

        create_agent_app(settings, checkpointer)

        create_llm.assert_called_once_with(settings)
        self.assertIs(build_graph.call_args.args[0], llm)
        self.assertIs(
            build_graph.call_args.kwargs["checkpointer"], checkpointer
        )
        self.assertIsNone(build_graph.call_args.kwargs["store"])

    @patch("builtins.print")
    @patch("builtins.input", side_effect=["你好", "exit"])
    def test_request_failure_is_handled_at_cli_boundary(self, _input, output):
        app = Mock()
        app.invoke.side_effect = RuntimeError("model unavailable")

        with self.assertLogs("ai_agent_learning.cli", level="ERROR"):
            run_cli(app, "user_001")

        output.assert_any_call("抱歉，本次请求执行失败，请稍后重试。")
        self.assertEqual(
            app.invoke.call_args.kwargs["config"],
            {"configurable": {"thread_id": "user_001"}},
        )

    @patch("builtins.print")
    @patch("builtins.input", side_effect=["第一轮", "第二轮", "exit"])
    def test_continuous_cli_session_reuses_thread_id(self, _input, _output):
        app = Mock()
        app.invoke.side_effect = [
            {"messages": [AIMessage(content="回答一")]},
            {"messages": [AIMessage(content="回答二")]},
        ]

        run_cli(app, "user_001")

        self.assertEqual(app.invoke.call_count, 2)
        for call in app.invoke.call_args_list:
            self.assertEqual(
                call.kwargs["config"],
                {"configurable": {"thread_id": "user_001"}},
            )

    @patch("ai_agent_learning.cli.show_state_history")
    @patch("ai_agent_learning.cli.show_current_state")
    @patch("builtins.input", side_effect=["/state", "/history", "exit"])
    def test_checkpoint_commands_do_not_invoke_agent(
        self,
        _input,
        show_current_state,
        show_state_history,
    ):
        app = Mock()

        run_cli(app, "user_001")

        show_current_state.assert_called_once_with(app, "user_001")
        show_state_history.assert_called_once_with(app, "user_001")
        app.invoke.assert_not_called()

    @patch("ai_agent_learning.cli.replay_checkpoint")
    @patch("ai_agent_learning.cli.validate_replay_checkpoint")
    @patch("ai_agent_learning.cli.show_state_history")
    @patch("builtins.input", return_value="1")
    def test_replay_command_selects_checkpoint_by_display_sequence(
        self,
        _input,
        show_state_history,
        validate_replay_checkpoint,
        replay_checkpoint,
    ):
        app = Mock()
        snapshot = Mock()
        snapshot.config = {
            "configurable": {
                "thread_id": "user_001",
                "checkpoint_id": "checkpoint-001",
            }
        }
        snapshot.next = ("tools",)
        show_state_history.return_value = [snapshot]
        replay_checkpoint.return_value = {
            "messages": [AIMessage(content="计算结果是 42")]
        }

        run_replay_command(app, "user_001")

        validate_replay_checkpoint.assert_called_once_with(snapshot, "user_001")
        replay_checkpoint.assert_called_once_with(
            app, snapshot, "user_001", context=None
        )

    @patch("ai_agent_learning.cli.fork_calculation_result")
    @patch("ai_agent_learning.cli.validate_fork_checkpoint")
    @patch("ai_agent_learning.cli.show_state_history")
    @patch("builtins.input", side_effect=["1", "43"])
    def test_fork_command_uses_selected_checkpoint_and_new_result(
        self,
        _input,
        show_state_history,
        validate_fork_checkpoint,
        fork_calculation_result,
    ):
        app = Mock()
        snapshot = Mock()
        snapshot.config = {
            "configurable": {
                "thread_id": "user_001",
                "checkpoint_id": "checkpoint-001",
            }
        }
        show_state_history.return_value = [snapshot]
        validate_fork_checkpoint.return_value = ToolMessage(
            content="42",
            tool_call_id="call-1",
            name="calculate",
            id="tool-message-1",
        )
        fork_calculation_result.return_value = (
            {
                "configurable": {
                    "thread_id": "user_001",
                    "checkpoint_id": "checkpoint-002",
                }
            },
            {"messages": [AIMessage(content="计算结果是 43")]},
        )

        run_fork_command(app, "user_001")

        validate_fork_checkpoint.assert_called_once_with(snapshot, "user_001")
        fork_calculation_result.assert_called_once_with(
            app,
            snapshot,
            "user_001",
            "43",
            context=None,
        )

    @patch("builtins.print")
    @patch("builtins.input", side_effect=["请记住，我喜欢Python", "approve", "exit"])
    def test_cli_approves_interrupt_with_command_resume(self, _input, _output):
        app = Mock()
        app.get_state.return_value.interrupts = ()
        app.invoke.side_effect = [
            {
                "messages": [AIMessage(content="")],
                "__interrupt__": [
                    Interrupt(
                        value={
                            "action": "save_user_memory",
                            "tool_name": "save_memory",
                            "arguments": {"content": "我喜欢Python"},
                            "message": "是否批准？",
                        },
                        id="interrupt-1",
                    )
                ],
            },
            {"messages": [AIMessage(content="记忆保存成功。")]},
        ]

        run_cli(app, "thread_hitl_001")

        resume_call = app.invoke.call_args_list[1]
        self.assertIsInstance(resume_call.args[0], Command)
        self.assertEqual(resume_call.args[0].resume, {"approved": True})
        self.assertEqual(
            resume_call.kwargs["config"],
            {"configurable": {"thread_id": "thread_hitl_001"}},
        )

    @patch("builtins.print")
    @patch("builtins.input", side_effect=["reject", "exit"])
    def test_cli_resumes_persisted_interrupt_after_restart(self, _input, _output):
        app = Mock()
        app.get_state.return_value.interrupts = (
            Interrupt(
                value={
                    "action": "save_user_memory",
                    "tool_name": "save_memory",
                    "arguments": {"content": "我喜欢Python"},
                    "message": "是否批准？",
                },
                id="interrupt-1",
            ),
        )
        app.invoke.return_value = {
            "messages": [AIMessage(content="记忆保存操作已取消。")]
        }

        run_cli(app, "thread_hitl_003")

        resume_command = app.invoke.call_args.args[0]
        self.assertIsInstance(resume_command, Command)
        self.assertEqual(
            resume_command.resume,
            {"approved": False, "reason": "用户拒绝"},
        )

    @patch("builtins.print")
    @patch("builtins.input", side_effect=["retry", "exit"])
    def test_cli_resumes_retry_review_with_command(self, _input, _output):
        app = Mock()
        app.get_state.return_value.interrupts = (
            Interrupt(
                value={
                    "action": "tool_failure_review",
                    "failed_node": "tools",
                    "error": "模拟超时",
                    "retry_count": 3,
                    "options": ["retry", "cancel"],
                },
                id="retry-interrupt-1",
            ),
        )
        app.invoke.return_value = {
            "messages": [AIMessage(content="人工重试后成功")]
        }

        run_cli(app, "retry_thread")

        resume_command = app.invoke.call_args.args[0]
        self.assertIsInstance(resume_command, Command)
        self.assertEqual(resume_command.resume, {"action": "retry"})
        self.assertEqual(
            app.invoke.call_args.kwargs["config"],
            {"configurable": {"thread_id": "retry_thread"}},
        )

    @patch("builtins.input", return_value="")
    def test_empty_session_id_uses_stable_default(self, _input):
        self.assertEqual(prompt_thread_id(), DEFAULT_THREAD_ID)

    @patch("builtins.input", return_value=" user_002 ")
    def test_session_id_is_read_once_at_startup(self, _input):
        self.assertEqual(prompt_thread_id(), "user_002")

    @patch("builtins.input", return_value="")
    def test_empty_user_id_uses_stable_default(self, _input):
        self.assertEqual(prompt_user_id(), DEFAULT_USER_ID)

    @patch("builtins.input", return_value=" user_001 ")
    def test_user_id_is_read_at_startup(self, _input):
        self.assertEqual(prompt_user_id(), "user_001")


if __name__ == "__main__":
    unittest.main()
    prompt_user_id,
