import unittest
from unittest.mock import Mock, patch

from ai_agent_learning.cli import run_cli


class CliTests(unittest.TestCase):
    @patch("builtins.print")
    @patch("builtins.input", side_effect=["你好", "exit"])
    def test_request_failure_is_handled_at_cli_boundary(self, _input, output):
        app = Mock()
        app.invoke.side_effect = RuntimeError("model unavailable")

        with self.assertLogs("ai_agent_learning.cli", level="ERROR"):
            run_cli(app)

        output.assert_any_call("抱歉，本次请求执行失败，请稍后重试。")


if __name__ == "__main__":
    unittest.main()
