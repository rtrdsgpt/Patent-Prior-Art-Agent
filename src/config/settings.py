"""Centralized settings via pydantic-settings, reading env vars / `.env`.

Same pattern as Financial Anomaly Detection Using RAG's `src/config/settings.py` — a cached
`Settings` singleton usable as a FastAPI dependency once the API layer exists.
"""

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM — comma-separated if multiple keys are available. Groq's free tier has fairly
    # tight per-key rate limits, and the agent pipeline (todo.md section 2) makes several
    # LLM calls per job across multiple stages, so `agents/groq_client.py` rotates across
    # all configured keys on a 429 rather than using only the first.
    groq_api_key: Optional[str] = None

    @property
    def groq_api_keys(self) -> list[str]:
        if not self.groq_api_key:
            return []
        return [key.strip() for key in self.groq_api_key.split(",") if key.strip()]

    # Confirmed available via a live `models.list()` call against the configured keys
    # before picking a default — see log.md.
    groq_model: str = "llama-3.3-70b-versatile"

    # BigQuery / Google Patents Public Data
    gcp_project_id: Optional[str] = None
    google_application_credentials: Optional[str] = None

    # Corpus scope — see docs/cpc_scope.md for why G06N3 was picked for the first slice
    target_cpc_class: str = "G06N3"
    # Cap on patents pulled per ingestion run — keeps BigQuery cost/time bounded for the
    # first working slice; see docs/cpc_scope.md's "scope discipline" section.
    corpus_size: int = 300
    # Where `ingestion/ingest_corpus.py` writes, and `ingestion/corpus.py` reads, the
    # ingested corpus cache — see corpus.py's docstring for why this is cached at all.
    corpus_cache_path: str = "data/corpus.json"

    # Retrieval
    vector_store_backend: str = "chroma"
    chroma_persist_directory: str = "chroma_db"
    # Set when Chroma runs as its own service (docker-compose) instead of embedded/local —
    # see docker-compose.yml and api/pipeline.py's _build_chroma_client.
    chroma_host: Optional[str] = None
    chroma_port: int = 8000
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
