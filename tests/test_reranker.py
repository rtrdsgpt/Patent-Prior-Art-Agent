import pytest

from patent_agent.config.settings import Settings
from patent_agent.ingestion.fixtures import load_fixture_patents
from patent_agent.retrieval.reranker import rerank
from patent_agent.schema import SearchResult

pytestmark = pytest.mark.slow  # loads a real local cross-encoder model

PATENTS = load_fixture_patents()
PATENTS_BY_ID = {p.patent_id: p for p in PATENTS}


def _all_patent_candidates() -> list[SearchResult]:
    return [SearchResult(patent_id=p.patent_id, score=0.5, retrieval_method="hybrid") for p in PATENTS]


def test_rerank_surfaces_true_positive_above_distractors():
    # The dropout patent (US10000001B2) should outrank patents about unrelated NN topics
    # (federated learning, attention) once the cross-encoder scores query-document jointly.
    results = rerank(
        "A technique for training a neural network that randomly disables neurons to avoid overfitting",
        _all_patent_candidates(),
        PATENTS_BY_ID,
        settings=Settings(rerank_top_k=8),
    )
    assert results[0].patent_id == "US10000001B2"


def test_rerank_respects_rerank_top_k():
    results = rerank(
        "neural network training",
        _all_patent_candidates(),
        PATENTS_BY_ID,
        settings=Settings(rerank_top_k=3),
    )
    assert len(results) == 3


def test_rerank_tags_retrieval_method():
    results = rerank(
        "neural network training",
        _all_patent_candidates(),
        PATENTS_BY_ID,
        settings=Settings(rerank_top_k=1),
    )
    assert results[0].retrieval_method == "reranked"


def test_rerank_scores_are_descending():
    results = rerank(
        "neural network training",
        _all_patent_candidates(),
        PATENTS_BY_ID,
        settings=Settings(rerank_top_k=8),
    )
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_rerank_skips_candidates_missing_from_patents_by_id():
    candidates = [SearchResult(patent_id="does-not-exist", score=1.0, retrieval_method="hybrid")]
    results = rerank("neural network", candidates, PATENTS_BY_ID, settings=Settings())
    assert results == []


def test_rerank_empty_candidates_returns_empty():
    results = rerank("neural network", [], PATENTS_BY_ID, settings=Settings())
    assert results == []
