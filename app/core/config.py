from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    OPENAI_API_KEY: str = "dummy-key"
    OPENAI_BASE_URL: Optional[str] = None
    LLM_MODEL: str = "gemma-4-E4B-it"
    LLM_TEMPERATURE: float = 0.0

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()

