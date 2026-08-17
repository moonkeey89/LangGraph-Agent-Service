import unittest
from unittest.mock import patch

from ai_agent_learning.config import Settings
from ai_agent_learning.llm import create_llm


class LlmFactoryTests(unittest.TestCase):
    @patch("ai_agent_learning.llm.ChatOpenAI")
    def test_create_llm_preserves_current_defaults(self, chat_openai):
        settings = Settings(
            _env_file=None,
            deepseek_api_key="test-secret-key",
        )

        create_llm(settings)

        chat_openai.assert_called_once_with(
            model="deepseek-chat",
            api_key="test-secret-key",
            base_url="https://api.deepseek.com",
        )

    @patch("ai_agent_learning.llm.ChatOpenAI")
    def test_create_llm_passes_configured_temperature(self, chat_openai):
        settings = Settings(
            _env_file=None,
            deepseek_api_key="test-secret-key",
            deepseek_temperature=0.3,
        )

        create_llm(settings)

        chat_openai.assert_called_once_with(
            model="deepseek-chat",
            api_key="test-secret-key",
            base_url="https://api.deepseek.com",
            temperature=0.3,
        )


if __name__ == "__main__":
    unittest.main()
