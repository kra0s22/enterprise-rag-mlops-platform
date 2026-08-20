"""Application configuration loaded from ``RAG_*`` environment variables."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration object for the whole platform.

    Every value can be overridden via environment variables prefixed with
    ``RAG_`` (see ``.env.example`` for the full reference).
    """

    model_config = SettingsConfigDict(env_file=".env", env_prefix="RAG_", extra="ignore")

    # Application
    app_name: str = "rag-platform"
    environment: str = "development"
    log_level: str = "INFO"

    # Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 32
    embedding_dimension: int = 384

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 64

    # Vector store backend: "qdrant" | "milvus"
    vector_store_provider: str = "qdrant"

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "documents"
    qdrant_api_key: str | None = None

    # Milvus
    milvus_uri: str = "http://localhost:19530"
    milvus_collection: str = "documents"
    milvus_token: str | None = None

    # Search
    default_top_k: int = 5
    score_threshold: float = 0.0

    # Generation (Ollama)
    ollama_url: str = "http://localhost:11434"
    llm_model: str = "llama3.2:3b"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 512

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "rag_platform"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
