"""What the `/disclosure/analyze` job actually runs today.

Retrieval (todo.md section 1) plus the first two section-2 agents — disclosure-parser and
search — are wired in now that Groq credentials are available. The claims-parser,
comparison, and risk-report agents aren't built yet, so a job still produces a ranked
candidate prior-art list, not a full `FTOReport`. The `/report/{job_id}` route reflects that
honestly (501) rather than fabricating a report.

Corpus comes from `ingestion.corpus.load_corpus()` — the real BigQuery-ingested cache if
one has been generated, else the fixture set. See that module's docstring for why the swap
between the two doesn't require changing anything here.
"""

from __future__ import annotations

from functools import lru_cache

import chromadb
from chromadb.api.models.Collection import Collection

from agents.disclosure_parser import parse_disclosure
from agents.search_agent import search_prior_art
from config.settings import Settings, get_settings
from ingestion.corpus import load_corpus
from retrieval.bm25_index import BM25Index, build_bm25_index
from retrieval.embedding_index import build_embedding_index
from schema import Patent, SearchResult


def _build_chroma_client(settings: Settings) -> chromadb.ClientAPI:
    """`chroma_host` is set when Chroma runs as its own docker-compose service; otherwise
    fall back to an on-disk persistent client so a local (non-Docker) run of the API
    doesn't re-embed the corpus on every process restart."""
    if settings.chroma_host:
        return chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
    return chromadb.PersistentClient(path=settings.chroma_persist_directory)


@lru_cache
def _get_indexes() -> tuple[BM25Index, Collection, dict[str, Patent]]:
    """Build both retrieval indexes once per process and reuse them across requests —
    embedding a corpus is real, non-trivial work, not something to redo per job."""
    settings = get_settings()
    patents = load_corpus(settings)
    bm25_index = build_bm25_index(patents)
    embedding_collection = build_embedding_index(patents, settings=settings, client=_build_chroma_client(settings))
    patents_by_id = {p.patent_id: p for p in patents}
    return bm25_index, embedding_collection, patents_by_id


def run_prior_art_search(disclosure_text: str) -> list[SearchResult]:
    """Parse the disclosure, then hybrid search + rerank (with adaptive query expansion)
    against the indexed corpus. Makes real Groq calls now that credentials exist — see
    `agents/disclosure_parser.py` and `agents/search_agent.py`.
    """
    settings = get_settings()
    bm25_index, embedding_collection, patents_by_id = _get_indexes()

    disclosure = parse_disclosure(disclosure_text, settings=settings)
    return search_prior_art(disclosure, bm25_index, embedding_collection, patents_by_id, settings=settings)
