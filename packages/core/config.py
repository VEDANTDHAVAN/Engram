from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

class Settings(BaseSettings):
    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------
    app_name: str = Field(default="Engram")
    environment: str = Field(default="development")
    debug: bool = Field(default=True)
    log_level: str = Field(default="INFO")

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)

    # ------------------------------------------------------------------
    # Database
    # ------------------------------------------------------------------
    database_url: str

    # ------------------------------------------------------------------
    # Redis (Optional for MVP)
    # ------------------------------------------------------------------
    redis_url: str | None = None

    # LLM

    LLM_PROVIDER: str = "groq"
    LLM_MODEL: str = "openai/gpt-oss-120b"
    GROQ_API_KEY: str
    JINA_API_KEY: str

    # Embeddings

    EMBEDDING_PROVIDER: str = "jina"
    EMBEDDING_MODEL: str = "jina-embeddings-v3"
    EMBEDDING_DIMENSIONS: int = 1024

    # Vector Retrieval
    
    VECTOR_PROVIDER: str = "turbovec"
    TURBOVEC_INDEX_PATH: str = (
        "data/turbovec/semantic_memory.tvim"
    )
    TURBOVEC_BIT_WIDTH: int = 4

    # Chunker
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64
    TOKENIZER_ENCODING: str = "cl100k_base"
    
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


@lru_cache
def get_settings() -> Settings:
    """
    Returns a cached Settings instance.

    The settings object is created only once during the application's
    lifetime and reused everywhere.
    """
    return Settings() # type: ignore


settings = get_settings()