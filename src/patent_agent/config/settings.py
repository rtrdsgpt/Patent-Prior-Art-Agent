"""Centralized settings via pydantic-settings, reading env vars / `.env`.

Same pattern as Financial Anomaly Detection Using RAG's `src/config/settings.py` — a cached
`Settings` singleton usable as a FastAPI dependency once the API layer exists.
"""

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM
    groq_api_key: Optional[str] = None

    # BigQuery / Google Patents Public Data
    gcp_project_id: Optional[str] = None
    google_application_credentials: Optional[str] = None

    # Corpus scope — see docs/cpc_scope.md for why G06N3 was picked for the first slice
    target_cpc_class: str = "G06N3"

    # Retrieval
    vector_store_backend: str = "chroma"
    chroma_persist_directory: str = "chroma_db"
    bm25_top_k: int = 50
    dense_top_k: int = 50
    hybrid_top_k: int = 20
    rerank_top_k: int = 10
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()
