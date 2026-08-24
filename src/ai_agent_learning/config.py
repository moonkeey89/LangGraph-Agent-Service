from pathlib import Path
from typing import Literal

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ai_agent_learning.knowledge.models import validate_knowledge_base_id


class Settings(BaseSettings):
    deepseek_api_key: SecretStr
    deepseek_model: str = "deepseek-chat"
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_temperature: float | None = None
    memory_embedding_model: str = "minishlab/potion-multilingual-128M"
    memory_embedding_dimensions: int = 256
    memory_manager_confidence_threshold: float = 0.75
    supervisor_max_subagent_calls: int = 4
    knowledge_base_id: str = "demo"
    knowledge_chroma_directory: Path = Path("data/knowledge_chroma")
    knowledge_catalog_path: Path = Path("data/knowledge_catalog.sqlite")
    knowledge_source_directory: Path = Path("data/knowledge_sources")
    knowledge_chunk_size: int = 800
    knowledge_chunk_overlap: int = 120
    knowledge_top_k: int = 3
    knowledge_relevance_threshold: float | None = 0.35
    knowledge_upload_max_file_size_mb: int = 10
    knowledge_upload_max_files: int = 5
    researchflow_database_path: Path = Path("data/researchflow.sqlite")
    auth_database_path: Path = Path("data/auth.sqlite")
    auth_session_cookie_name: str = "researchflow_session"
    auth_csrf_cookie_name: str = "researchflow_csrf"
    auth_csrf_header_name: str = "X-CSRF-Token"
    auth_session_ttl_minutes: int = 480
    auth_cookie_secure: bool = False
    auth_cookie_samesite: Literal["lax", "strict", "none"] = "lax"
    auth_cookie_domain: str | None = None
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

    @field_validator("knowledge_base_id")
    @classmethod
    def validate_default_knowledge_base_id(cls, value: str) -> str:
        return validate_knowledge_base_id(value)

    @field_validator(
        "knowledge_chunk_size",
        "knowledge_top_k",
        "knowledge_upload_max_file_size_mb",
        "knowledge_upload_max_files",
    )
    @classmethod
    def validate_positive_knowledge_integer(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("知识库chunk_size和top_k必须是正整数")
        return value

    @field_validator("knowledge_chunk_overlap")
    @classmethod
    def validate_knowledge_overlap(cls, value: int) -> int:
        if value < 0:
            raise ValueError("knowledge_chunk_overlap不能为负数")
        return value

    @field_validator("knowledge_relevance_threshold")
    @classmethod
    def validate_knowledge_threshold(
        cls,
        value: float | None,
    ) -> float | None:
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError("knowledge_relevance_threshold必须在0到1之间")
        return value

    def model_post_init(self, _context) -> None:
        if self.knowledge_chunk_overlap >= self.knowledge_chunk_size:
            raise ValueError(
                "knowledge_chunk_overlap必须小于knowledge_chunk_size"
            )
        if self.auth_cookie_samesite == "none" and not self.auth_cookie_secure:
            raise ValueError("SameSite=None 时必须启用 Secure Cookie")

    @field_validator("auth_session_ttl_minutes")
    @classmethod
    def validate_auth_session_ttl(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("认证Session有效期必须是正整数")
        return value

    @field_validator(
        "auth_session_cookie_name",
        "auth_csrf_cookie_name",
        "auth_csrf_header_name",
    )
    @classmethod
    def validate_auth_names(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("认证Cookie和Header名称不能为空")
        return normalized

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
