from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = Field(
        default="postgresql+psycopg://mtg:mtg@localhost:5432/mtg_deck_builder",
        alias="DATABASE_URL",
    )
    backend_cors_origins: str = Field(
        default="http://localhost:5173,http://127.0.0.1:5173,http://0.0.0.0:5173",
        alias="BACKEND_CORS_ORIGINS",
    )
    scryfall_api_base_url: str = Field(default="https://api.scryfall.com", alias="SCRYFALL_API_BASE_URL")
    embedding_dimensions: int = Field(default=384, alias="EMBEDDING_DIMENSIONS")
    embedding_provider: str = Field(default="fastembed", alias="EMBEDDING_PROVIDER")
    embedding_model: str = Field(default="BAAI/bge-small-en-v1.5", alias="EMBEDDING_MODEL")
    embedding_cache_enabled: bool = Field(default=True, alias="EMBEDDING_CACHE_ENABLED")
    embedding_cache_path: str = Field(default="data/embedding-cache.sqlite3", alias="EMBEDDING_CACHE_PATH")
    embedding_model_cache_path: str = Field(default="data/fastembed-cache", alias="EMBEDDING_MODEL_CACHE_PATH")
    rag_ingest_batch_size: int = Field(default=100, alias="RAG_INGEST_BATCH_SIZE")
    rag_ingest_batch_delay_seconds: float = Field(default=0.0, alias="RAG_INGEST_BATCH_DELAY_SECONDS")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_base_url: str = Field(default="https://api.openai.com/v1", alias="OPENAI_BASE_URL")
    openai_model: str = Field(default="gpt-5.4-mini", alias="OPENAI_MODEL")
    agent_provider: str = Field(default="openai", alias="AGENT_PROVIDER")
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="llama3.2", alias="OLLAMA_MODEL")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_format: str = Field(default="plain", alias="LOG_FORMAT")

    model_config = SettingsConfigDict(env_file=("../.env", ".env"), extra="ignore")

    @property
    def cors_origins(self) -> list[str]:
        origins = [origin.strip() for origin in self.backend_cors_origins.split(",") if origin.strip()]
        if "*" in origins:
            return ["*"]
        return origins

    @property
    def openai_enabled(self) -> bool:
        return bool(self.openai_api_key.strip())

    @property
    def ollama_enabled(self) -> bool:
        return self.agent_provider.strip().lower() == "ollama"

    @property
    def openai_agent_enabled(self) -> bool:
        return self.agent_provider.strip().lower() == "openai" and self.openai_enabled


@lru_cache
def get_settings() -> Settings:
    return Settings()
