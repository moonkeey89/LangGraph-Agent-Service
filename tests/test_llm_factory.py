import unittest
from unittest.mock import patch

from llm_factory import create_llm
from settings import Settings


class LlmFactoryTests(unittest.TestCase):
    @patch("llm_factory.ChatOpenAI")
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

    @patch("llm_factory.ChatOpenAI")
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
