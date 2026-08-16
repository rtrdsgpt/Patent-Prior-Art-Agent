"""What the `/disclosure/analyze` job actually runs today.

Only the retrieval half of todo.md's pipeline exists right now: hybrid search + reranking
over the indexed corpus (todo.md section 1). The disclosure-parser, comparison, and
risk-report agents (section 2) are paused pending a Groq API key (see log.md) — so a job
currently produces a ranked candidate prior-art list, not a full `FTOReport`. The
`/report/{job_id}` route reflects that honestly (501) rather than fabricating a report.

Corpus is the fixture set (`ingestion.fixtures`) until real BigQuery ingestion lands — see
`fixtures.py`'s docstring for why that swap won't require changing this module.
"""

from __future__ import annotations

from functools import lru_cache

from chromadb.api.models.Collection import Collection

from patent_agent.config.settings import get_settings
from patent_agent.ingestion.fixtures import load_fixture_patents
from patent_agent.retrieval.bm25_index import BM25Index, build_bm25_index
from patent_agent.retrieval.embedding_index import build_embedding_index
from patent_agent.retrieval.hybrid import hybrid_search
from patent_agent.retrieval.reranker import rerank
from patent_agent.schema import Patent, SearchResult


@lru_cache
def _get_indexes() -> tuple[BM25Index, Collection, dict[str, Patent]]:
    """Build both retrieval indexes once per process and reuse them across requests —
    embedding a corpus is real, non-trivial work, not something to redo per job."""
    patents = load_fixture_patents()
    bm25_index = build_bm25_index(patents)
    embedding_collection = build_embedding_index(patents)
    patents_by_id = {p.patent_id: p for p in patents}
    return bm25_index, embedding_collection, patents_by_id


def run_prior_art_search(disclosure_text: str) -> list[SearchResult]:
    """Hybrid search + rerank the disclosure's raw text against the indexed corpus.

    Uses the raw disclosure text directly as the query, since the disclosure-parser agent
    that would extract structured elements/candidate CPC classes from it doesn't exist yet.
    Real once agents land: the retrieval call itself doesn't change, only what query text
    feeds into it.
    """
    settings = get_settings()
    bm25_index, embedding_collection, patents_by_id = _get_indexes()

    candidates = hybrid_search(bm25_index, embedding_collection, disclosure_text, settings=settings)
    return rerank(disclosure_text, candidates, patents_by_id, settings=settings)
