from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    deepseek_api_key: SecretStr
    deepseek_model: str = "deepseek-chat"
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_temperature: float | None = None
    memory_embedding_model: str = "minishlab/potion-multilingual-128M"
    memory_embedding_dimensions: int = 256
    log_level: str = "INFO"

    @field_validator("memory_embedding_dimensions")
    @classmethod
    def validate_embedding_dimensions(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("Embedding 维度必须是正整数")
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
