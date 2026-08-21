from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    deepseek_api_key: SecretStr
    deepseek_model: str = "deepseek-chat"
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_temperature: float | None = None
    memory_embedding_model: str = "minishlab/potion-multilingual-128M"
    memory_embedding_dimensions: int = 256
    memory_manager_confidence_threshold: float = 0.75
    supervisor_max_subagent_calls: int = 4
    log_level: str = "INFO"

    @field_validator("memory_embedding_dimensions")
    @classmethod
    def validate_embedding_dimensions(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("Embedding 维度必须是正整数")
        return value

    @field_validator("memory_manager_confidence_threshold")
    @classmethod
    def validate_memory_confidence_threshold(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("Memory Manager 置信度阈值必须在 0 到 1 之间")
        return value

    @field_validator("supervisor_max_subagent_calls")
    @classmethod
    def validate_max_subagent_calls(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("Supervisor 最大 Subagent 调用次数必须是正整数")
        return value

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("log_level")
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        normalized = value.upper()
        valid_levels = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}

        if normalized not in valid_levels:
            raise ValueError(f"日志级别必须是以下值之一：{sorted(valid_levels)}")

        return normalized
