"""Combines BM25 and dense retrieval results into one ranked candidate list.

Uses Reciprocal Rank Fusion (RRF) rather than combining raw scores: BM25 scores and dense
cosine-style similarity scores live on different, uncalibrated scales, so summing or
averaging them directly would let whichever ranker happens to produce larger numbers
dominate. RRF only looks at each ranker's *rank order*, which sidesteps that entirely — a
patent ranked #1 by BM25 and #1 by dense search wins regardless of the two rankers' raw
score magnitudes.
"""

from __future__ import annotations

from chromadb.api.models.Collection import Collection

from patent_agent.config.settings import Settings, get_settings
from patent_agent.retrieval.bm25_index import BM25Index, bm25_search
from patent_agent.retrieval.embedding_index import dense_search
from patent_agent.schema import SearchResult

_DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    result_lists: list[list[SearchResult]], top_k: int, rrf_k: int = _DEFAULT_RRF_K
) -> list[SearchResult]:
    """Fuse multiple ranked result lists via RRF: score(patent) = sum(1 / (rrf_k + rank)).

    `rrf_k` dampens the influence of low ranks (a common literature default is 60); a
    patent absent from a given list simply doesn't get that list's contribution, rather
    than being penalized further.
    """
    scores: dict[str, float] = {}
    for results in result_lists:
        for rank, result in enumerate(results, start=1):
            scores[result.patent_id] = scores.get(result.patent_id, 0.0) + 1.0 / (rrf_k + rank)

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [
        SearchResult(patent_id=patent_id, score=score, retrieval_method="hybrid")
        for patent_id, score in ranked[:top_k]
    ]


def hybrid_search(
    bm25_index: BM25Index,
    embedding_collection: Collection,
    query: str,
    settings: Settings | None = None,
) -> list[SearchResult]:
    """Run BM25 and dense search independently, then fuse via RRF.

    Each ranker is over-fetched to its own `*_top_k` (see `Settings`) before fusion narrows
    to `hybrid_top_k` — fusing from a wider candidate pool than the final result count is
    what lets a patent that one ranker alone wouldn't have surfaced near the top still
    contribute to the fused ranking.
    """
    settings = settings or get_settings()
    bm25_results = bm25_search(bm25_index, query, top_k=settings.bm25_top_k)
    dense_results = dense_search(embedding_collection, query, top_k=settings.dense_top_k)
    return reciprocal_rank_fusion([bm25_results, dense_results], top_k=settings.hybrid_top_k)
