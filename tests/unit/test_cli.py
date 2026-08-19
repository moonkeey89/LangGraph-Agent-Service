import unittest
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage

from ai_agent_learning.cli import (
    DEFAULT_THREAD_ID,
    create_agent_app,
    prompt_thread_id,
    run_cli,
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

    @patch("builtins.input", return_value="")
    def test_empty_session_id_uses_stable_default(self, _input):
        self.assertEqual(prompt_thread_id(), DEFAULT_THREAD_ID)

    @patch("builtins.input", return_value=" user_002 ")
    def test_session_id_is_read_once_at_startup(self, _input):
        self.assertEqual(prompt_thread_id(), "user_002")


if __name__ == "__main__":
    unittest.main()
