"""ml-api settings."""

from __future__ import annotations

from functools import lru_cache
from typing import ClassVar

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_prefix="ML_API_",
        extra="ignore",
    )

    api_v1_prefix: str = "/api/v1"
    worker_stream_origin: str = "http://127.0.0.1:8090"
    worker_stream_timeout_s: float = 3.0
    worker_probe_origin: str = ""
    worker_probe_timeout_s: float = 5.0
    connection_test_timeout_s: float = 5.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
