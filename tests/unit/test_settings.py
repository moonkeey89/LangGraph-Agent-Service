import os
import unittest
from unittest.mock import patch

from pydantic import ValidationError

from ai_agent_learning.config import Settings


class SettingsTests(unittest.TestCase):
    def test_defaults_preserve_current_deepseek_configuration(self):
        settings = Settings(
            _env_file=None,
            deepseek_api_key="test-secret-key",
        )

        self.assertEqual(settings.deepseek_model, "deepseek-chat")
        self.assertEqual(settings.deepseek_base_url, "https://api.deepseek.com")
        self.assertIsNone(settings.deepseek_temperature)
        self.assertEqual(settings.log_level, "INFO")
        self.assertNotIn("test-secret-key", repr(settings))

    def test_environment_can_override_configuration(self):
        environment = {
            "DEEPSEEK_API_KEY": "environment-secret-key",
            "DEEPSEEK_MODEL": "configured-model",
            "DEEPSEEK_BASE_URL": "https://example.com",
            "DEEPSEEK_TEMPERATURE": "0.2",
            "LOG_LEVEL": "debug",
        }

        with patch.dict(os.environ, environment, clear=True):
            settings = Settings(_env_file=None)

        self.assertEqual(
            settings.deepseek_api_key.get_secret_value(),
            "environment-secret-key",
        )
        self.assertEqual(settings.deepseek_model, "configured-model")
        self.assertEqual(settings.deepseek_base_url, "https://example.com")
        self.assertEqual(settings.deepseek_temperature, 0.2)
        self.assertEqual(settings.log_level, "DEBUG")

    def test_api_key_is_required(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValidationError):
                Settings(_env_file=None)

    def test_log_level_must_be_valid(self):
        with self.assertRaises(ValidationError):
            Settings(
                _env_file=None,
                deepseek_api_key="test-secret-key",
                log_level="verbose",
            )


if __name__ == "__main__":
    unittest.main()
