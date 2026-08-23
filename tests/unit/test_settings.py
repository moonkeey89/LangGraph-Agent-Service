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
        self.assertEqual(
            settings.memory_embedding_model,
            "minishlab/potion-multilingual-128M",
        )
        self.assertEqual(settings.memory_embedding_dimensions, 256)
        self.assertEqual(settings.memory_manager_confidence_threshold, 0.75)
        self.assertEqual(settings.supervisor_max_subagent_calls, 4)
        self.assertEqual(settings.knowledge_base_id, "demo")
        self.assertEqual(
            settings.knowledge_chroma_directory.as_posix(),
            "data/knowledge_chroma",
        )
        self.assertEqual(settings.knowledge_chunk_size, 800)
        self.assertEqual(settings.knowledge_chunk_overlap, 120)
        self.assertEqual(settings.knowledge_top_k, 3)
        self.assertEqual(settings.knowledge_relevance_threshold, 0.35)
        self.assertEqual(
            settings.knowledge_catalog_path.as_posix(),
            "data/knowledge_catalog.sqlite",
        )
        self.assertEqual(
            settings.knowledge_source_directory.as_posix(),
            "data/knowledge_sources",
        )
        self.assertEqual(settings.knowledge_upload_max_file_size_mb, 10)
        self.assertEqual(settings.knowledge_upload_max_files, 5)
        self.assertEqual(
            settings.researchflow_database_path.as_posix(),
            "data/researchflow.sqlite",
        )
        self.assertEqual(settings.log_level, "INFO")
        self.assertNotIn("test-secret-key", repr(settings))

    def test_environment_can_override_configuration(self):
        environment = {
            "DEEPSEEK_API_KEY": "environment-secret-key",
            "DEEPSEEK_MODEL": "configured-model",
            "DEEPSEEK_BASE_URL": "https://example.com",
            "DEEPSEEK_TEMPERATURE": "0.2",
            "MEMORY_EMBEDDING_MODEL": "test/embedding-model",
            "MEMORY_EMBEDDING_DIMENSIONS": "128",
            "MEMORY_MANAGER_CONFIDENCE_THRESHOLD": "0.8",
            "SUPERVISOR_MAX_SUBAGENT_CALLS": "5",
            "KNOWLEDGE_BASE_ID": "internal_docs",
            "KNOWLEDGE_CHROMA_DIRECTORY": "custom/chroma",
            "KNOWLEDGE_CHUNK_SIZE": "600",
            "KNOWLEDGE_CHUNK_OVERLAP": "80",
            "KNOWLEDGE_TOP_K": "4",
            "KNOWLEDGE_RELEVANCE_THRESHOLD": "0.45",
            "KNOWLEDGE_CATALOG_PATH": "custom/catalog.sqlite",
            "KNOWLEDGE_SOURCE_DIRECTORY": "custom/sources",
            "KNOWLEDGE_UPLOAD_MAX_FILE_SIZE_MB": "8",
            "KNOWLEDGE_UPLOAD_MAX_FILES": "3",
            "RESEARCHFLOW_DATABASE_PATH": "custom/research.sqlite",
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
        self.assertEqual(settings.memory_embedding_model, "test/embedding-model")
        self.assertEqual(settings.memory_embedding_dimensions, 128)
        self.assertEqual(settings.memory_manager_confidence_threshold, 0.8)
        self.assertEqual(settings.supervisor_max_subagent_calls, 5)
        self.assertEqual(settings.knowledge_base_id, "internal_docs")
        self.assertEqual(settings.knowledge_chunk_size, 600)
        self.assertEqual(settings.knowledge_chunk_overlap, 80)
        self.assertEqual(settings.knowledge_top_k, 4)
        self.assertEqual(settings.knowledge_relevance_threshold, 0.45)
        self.assertEqual(settings.knowledge_upload_max_file_size_mb, 8)
        self.assertEqual(settings.knowledge_upload_max_files, 3)
        self.assertEqual(
            settings.researchflow_database_path.as_posix(),
            "custom/research.sqlite",
        )
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

    def test_embedding_dimensions_must_be_positive(self):
        with self.assertRaises(ValidationError):
            Settings(
                _env_file=None,
                deepseek_api_key="test-secret-key",
                memory_embedding_dimensions=0,
            )

    def test_memory_confidence_threshold_must_be_between_zero_and_one(self):
        with self.assertRaises(ValidationError):
            Settings(
                _env_file=None,
                deepseek_api_key="test-secret-key",
                memory_manager_confidence_threshold=1.1,
            )

    def test_supervisor_max_subagent_calls_must_be_positive(self):
        with self.assertRaises(ValidationError):
            Settings(
                _env_file=None,
                deepseek_api_key="test-secret-key",
                supervisor_max_subagent_calls=0,
            )

    def test_knowledge_configuration_is_validated(self):
        invalid_values = [
            {"knowledge_base_id": "bad id"},
            {"knowledge_chunk_size": 0},
            {"knowledge_chunk_size": 100, "knowledge_chunk_overlap": 100},
            {"knowledge_top_k": 0},
            {"knowledge_relevance_threshold": 1.1},
            {"knowledge_upload_max_file_size_mb": 0},
            {"knowledge_upload_max_files": 0},
        ]
        for values in invalid_values:
            with self.subTest(values=values):
                with self.assertRaises(ValidationError):
                    Settings(
                        _env_file=None,
                        deepseek_api_key="test-secret-key",
                        **values,
                    )


if __name__ == "__main__":
    unittest.main()
