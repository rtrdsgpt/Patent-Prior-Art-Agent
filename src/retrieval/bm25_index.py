"""Sparse (lexical) retrieval over the claim-level chunk index, using BM25.

Complements `embedding_index.py`'s dense retrieval — BM25 catches exact terminology matches
(specific component names, chemical/technical terms) that a dense embedding can blur across
semantically-similar-but-lexically-different claims, and vice versa. `hybrid.py` combines
both. See todo.md section 1 ("Hybrid retrieval: BM25 + embeddings") for why both exist
rather than picking one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from ingestion.chunking import claim_to_index_chunk
from schema import Patent, SearchResult

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.lower())


@dataclass
class BM25Index:
    """Bundles the fitted BM25 model with the per-chunk patent_id it was built from —
    BM25Okapi itself only knows token lists, not what patent each one came from.

    `model` is `None` for an empty corpus: `BM25Okapi.__init__` divides by the corpus size
    to compute the average document length, which raises `ZeroDivisionError` on an empty
    corpus rather than handling it, so that case is special-cased here instead.
    """

    model: BM25Okapi | None
    patent_ids: list[str]


def build_bm25_index(patents: list[Patent]) -> BM25Index:
    patent_ids: list[str] = []
    tokenized_chunks: list[list[str]] = []

    for patent in patents:
        for claim in patent.claims:
            chunk = claim_to_index_chunk(claim, patent.title)
            patent_ids.append(patent.patent_id)
            tokenized_chunks.append(_tokenize(chunk))

    model = BM25Okapi(tokenized_chunks) if tokenized_chunks else None
    return BM25Index(model=model, patent_ids=patent_ids)


def bm25_search(index: BM25Index, query: str, top_k: int) -> list[SearchResult]:
    """Query the BM25 index, returning one `SearchResult` per patent (best-scoring claim
    chunk wins) — same per-patent dedup rule as `embedding_index.dense_search`, so the two
    result lists are directly comparable/combinable in `hybrid.py`."""
    if index.model is None:
        return []

    scores = index.model.get_scores(_tokenize(query))

    best_by_patent: dict[str, float] = {}
    for patent_id, score in zip(index.patent_ids, scores):
        if patent_id not in best_by_patent or score > best_by_patent[patent_id]:
            best_by_patent[patent_id] = score

    ranked = sorted(best_by_patent.items(), key=lambda kv: kv[1], reverse=True)
    return [
        SearchResult(patent_id=patent_id, score=score, retrieval_method="bm25")
        for patent_id, score in ranked[:top_k]
        if score > 0
    ]
