"""What the `/disclosure/analyze` job actually runs today: the full section-2 pipeline
(`agents/orchestrator.py`), now that all five agents exist and Groq credentials are
available. `POST /disclosure/analyze` produces a real `FTOReport`, not a placeholder.

Corpus comes from `ingestion.corpus.load_corpus()` — the real BigQuery-ingested cache if
one has been generated, else the fixture set. See that module's docstring for why the swap
between the two doesn't require changing anything here.
"""

from __future__ import annotations

from functools import lru_cache

import chromadb
from chromadb.api.models.Collection import Collection

from agents.citation_guard import verify_citations
from agents.claims_parser import parse_claim_elements
from agents.comparison_agent import assess_novelty
from agents.disclosure_parser import parse_disclosure
from agents.orchestrator import run_fto_pipeline
from agents.search_agent import search_prior_art
from config.settings import Settings, get_settings
from ingestion.corpus import load_corpus
from retrieval.bm25_index import BM25Index, build_bm25_index
from retrieval.embedding_index import build_embedding_index
from schema import FTOReport, NoveltyAssessment, Patent, SearchResult


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
    against the indexed corpus — used by the MCP `search_prior_art` tool
    (`mcp_server.py`), which only wants candidates, not a full report.
    """
    settings = get_settings()
    bm25_index, embedding_collection, patents_by_id = _get_indexes()

    disclosure = parse_disclosure(disclosure_text, settings=settings)
    return search_prior_art(disclosure, bm25_index, embedding_collection, patents_by_id, settings=settings)


def run_fto_analysis(disclosure_text: str) -> FTOReport:
    """Run the full pipeline — disclosure-parser through risk-report — and return a
    complete `FTOReport`. What `POST /disclosure/analyze` actually runs as a job.
    """
    bm25_index, embedding_collection, patents_by_id = _get_indexes()
    return run_fto_pipeline(disclosure_text, bm25_index, embedding_collection, patents_by_id, settings=get_settings())


def run_novelty_assessment(disclosure_text: str, candidate_patent_id: str) -> NoveltyAssessment:
    """Assess one specific candidate patent against a disclosure — used by the MCP
    `assess_novelty` tool (`mcp_server.py`), which names a single candidate rather than
    running a full search + multi-candidate report.

    Raises `KeyError` if `candidate_patent_id` isn't in the indexed corpus — the caller
    (the MCP tool) is responsible for turning that into a client-facing error.
    """
    settings = get_settings()
    _, _, patents_by_id = _get_indexes()
    patent = patents_by_id[candidate_patent_id]

    disclosure = parse_disclosure(disclosure_text, settings=settings)
    claim_elements = parse_claim_elements(patent, settings=settings)
    assessment = assess_novelty(disclosure, patent, claim_elements=claim_elements, settings=settings)
    return verify_citations(assessment, patent)
