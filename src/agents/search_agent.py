"""Prior-art search agent (todo.md section 2): hybrid search + rerank over the indexed
corpus, built from the disclosure-parser's structured output, with adaptive query expansion
if the reranked results look too thin or too generic.

Reuses the existing retrieval stack (`retrieval/hybrid.py`, `retrieval/reranker.py`)
unchanged — this agent's job is query *construction* and a bounded retry loop around it, not
new retrieval mechanics.
"""

from __future__ import annotations

import logging

from chromadb.api.models.Collection import Collection

from config.settings import Settings, get_settings
from retrieval.bm25_index import BM25Index
from retrieval.hybrid import hybrid_search
from retrieval.reranker import rerank
from schema import InventionDisclosure, Patent, SearchResult
from tracing import traced

logger = logging.getLogger(__name__)

# Reranked scores are cross-encoder logit-style (roughly -12 to +10 in practice — see
# reranker.py's slow tests for observed ranges), so 0 is a meaningful "plausibly relevant"
# cutoff, not an arbitrary one. MIN/MAX bound how many of the (at most `rerank_top_k`)
# results should clear that bar before the query is judged too narrow/too broad. These are
# heuristic starting points, not tuned against ground truth yet — section 5's recall@k eval
# is what would validate or adjust them once it exists.
_RELEVANCE_THRESHOLD = 0.0
_MIN_RELEVANT = 1
_MAX_RELEVANT = 8


def _relevant_count(results: list[SearchResult]) -> int:
    return sum(1 for r in results if r.score > _RELEVANCE_THRESHOLD)


def _candidate_queries(disclosure: InventionDisclosure) -> list[str]:
    """Three queries in a fixed, bounded sequence — not generated on the fly — so this
    agent never needs another LLM call (and its own retry/failure mode) just to decide what
    to search for next:

    1. medium — technical_field + all key_elements: the default, most-informed query.
    2. broad — technical_field alone: for when (1) came back too thin, dropping the
       (possibly overly specific) key_elements that may not share vocabulary with the corpus.
    3. narrow — technical_field + only the first two key_elements: for when (1) came back
       with too many plausible hits, on the theory the first elements extracted are usually
       the most central/specific ones.
    """
    elements_text = " ".join(disclosure.key_elements)
    return [
        f"{disclosure.technical_field}. {elements_text}".strip(". "),
        disclosure.technical_field or disclosure.raw_text,
        f"{disclosure.technical_field}. {' '.join(disclosure.key_elements[:2])}".strip(". "),
    ]


@traced("search_agent")
def search_prior_art(
    disclosure: InventionDisclosure,
    bm25_index: BM25Index,
    embedding_collection: Collection,
    patents_by_id: dict[str, Patent],
    settings: Settings | None = None,
) -> list[SearchResult]:
    """Hybrid search + rerank against the indexed corpus, adaptively retrying with a
    broader or narrower query if the first attempt's results look too thin or too generic.

    Bounded to the fixed 3-query sequence from `_candidate_queries` — reuses the same
    bounded-retry discipline as `agents/groq_client.py`'s key rotation and
    `agents/disclosure_parser.py`'s retry-on-malformed-JSON: try a fixed, small set of
    alternatives, then stop and return the best attempt rather than looping indefinitely.
    """
    settings = settings or get_settings()
    queries = _candidate_queries(disclosure)

    best_results: list[SearchResult] = []
    best_relevant_count = -1

    for attempt, query in enumerate(queries):
        candidates = hybrid_search(bm25_index, embedding_collection, query, settings=settings)
        results = rerank(query, candidates, patents_by_id, settings=settings)
        relevant = _relevant_count(results)

        logger.info("Search attempt %d/%d: query=%r -> %d/%d plausibly relevant", attempt + 1, len(queries), query, relevant, len(results))

        if relevant > best_relevant_count:
            best_results, best_relevant_count = results, relevant

        if _MIN_RELEVANT <= relevant <= _MAX_RELEVANT:
            return results

    return best_results
