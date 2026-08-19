import unittest

from ai_agent_learning.agent import (
    PermanentToolError,
    SideEffectUnknownError,
    classify_tool_error,
)


class RateLimitError(Exception):
    pass


class AuthenticationError(Exception):
    pass


class ErrorClassificationTests(unittest.TestCase):
    def test_error_categories_are_classified_conservatively(self):
        cases = [
            (TimeoutError("timeout"), "transient"),
            (ConnectionError("connection"), "transient"),
            (RateLimitError("limited"), "transient"),
            (ValueError("bad argument"), "invalid_arguments"),
            (PermissionError("denied"), "permission"),
            (AuthenticationError("bad key"), "permission"),
            (PermanentToolError("gone"), "permanent"),
            (SideEffectUnknownError("unknown"), "side_effect_unknown"),
            (RuntimeError("unknown exception"), "permanent"),
        ]

        for error, expected in cases:
            with self.subTest(error=type(error).__name__):
                self.assertEqual(classify_tool_error(error), expected)


if __name__ == "__main__":
    unittest.main()
