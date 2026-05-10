from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = Field(
        default="postgresql+psycopg://mtg:mtg@localhost:5432/mtg_deck_builder",
        alias="DATABASE_URL",
    )
    backend_cors_origins: str = Field(default="http://localhost:5173", alias="BACKEND_CORS_ORIGINS")
    scryfall_api_base_url: str = Field(default="https://api.scryfall.com", alias="SCRYFALL_API_BASE_URL")
    embedding_dimensions: int = Field(default=1536, alias="EMBEDDING_DIMENSIONS")
    embedding_provider: str = Field(default="hash", alias="EMBEDDING_PROVIDER")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.backend_cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
