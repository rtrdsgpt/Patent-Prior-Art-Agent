from unittest.mock import MagicMock

import pytest

from agents.search_agent import _candidate_queries, search_prior_art
from ingestion.fixtures import load_fixture_patents
from retrieval.bm25_index import build_bm25_index
from retrieval.embedding_index import build_embedding_index
from schema import InventionDisclosure, SearchResult


def _disclosure(**overrides) -> InventionDisclosure:
    fields = dict(
        raw_text="raw disclosure text",
        technical_field="neural network training regularization",
        key_elements=["dropout regularization", "backpropagation", "gradient descent"],
        candidate_cpc_classes=["G06N3/08"],
    )
    fields.update(overrides)
    return InventionDisclosure(**fields)


def _results(scores: list[float]) -> list[SearchResult]:
    return [SearchResult(patent_id=f"P{i}", score=s, retrieval_method="reranked") for i, s in enumerate(scores)]


def test_candidate_queries_medium_includes_field_and_all_elements():
    queries = _candidate_queries(_disclosure())
    assert "neural network training regularization" in queries[0]
    assert "dropout regularization" in queries[0]
    assert "gradient descent" in queries[0]


def test_candidate_queries_broad_is_technical_field_only():
    queries = _candidate_queries(_disclosure())
    assert queries[1] == "neural network training regularization"


def test_candidate_queries_narrow_uses_only_first_two_elements():
    queries = _candidate_queries(_disclosure())
    assert "dropout regularization" in queries[2]
    assert "backpropagation" in queries[2]
    assert "gradient descent" not in queries[2]


def test_search_prior_art_returns_first_attempt_when_relevance_in_range(monkeypatch):
    monkeypatch.setattr("agents.search_agent.hybrid_search", MagicMock(return_value=[]))
    rerank_mock = MagicMock(return_value=_results([5.0, 3.0, -1.0]))  # 2 relevant, within [1, 8]
    monkeypatch.setattr("agents.search_agent.rerank", rerank_mock)

    search_prior_art(_disclosure(), bm25_index=MagicMock(), embedding_collection=MagicMock(), patents_by_id={})

    assert rerank_mock.call_count == 1


def test_search_prior_art_retries_with_broad_query_when_too_few_relevant(monkeypatch):
    monkeypatch.setattr("agents.search_agent.hybrid_search", MagicMock(return_value=[]))
    rerank_mock = MagicMock(side_effect=[_results([-1.0, -2.0]), _results([4.0])])  # 0 relevant, then 1 relevant
    monkeypatch.setattr("agents.search_agent.rerank", rerank_mock)

    result = search_prior_art(_disclosure(), bm25_index=MagicMock(), embedding_collection=MagicMock(), patents_by_id={})

    assert rerank_mock.call_count == 2
    assert result == _results([4.0])


def test_search_prior_art_retries_with_narrow_query_when_too_many_relevant(monkeypatch):
    monkeypatch.setattr("agents.search_agent.hybrid_search", MagicMock(return_value=[]))
    too_many = _results([5.0] * 9)  # 9 relevant, above MAX_RELEVANT=8
    just_right = _results([5.0, 5.0])  # 2 relevant
    rerank_mock = MagicMock(side_effect=[too_many, just_right])
    monkeypatch.setattr("agents.search_agent.rerank", rerank_mock)

    result = search_prior_art(_disclosure(), bm25_index=MagicMock(), embedding_collection=MagicMock(), patents_by_id={})

    assert rerank_mock.call_count == 2
    assert result == just_right


def test_search_prior_art_returns_best_attempt_when_all_three_miss_the_range(monkeypatch):
    monkeypatch.setattr("agents.search_agent.hybrid_search", MagicMock(return_value=[]))
    attempt_1 = _results([-1.0])  # 0 relevant
    attempt_2 = _results([-1.0, -2.0])  # 0 relevant
    attempt_3 = _results([3.0, -1.0])  # 1 relevant -- best of the three, still returned even though in-range would've short-circuited
    rerank_mock = MagicMock(side_effect=[attempt_1, attempt_2, attempt_3])
    monkeypatch.setattr("agents.search_agent.rerank", rerank_mock)

    result = search_prior_art(_disclosure(), bm25_index=MagicMock(), embedding_collection=MagicMock(), patents_by_id={})

    assert rerank_mock.call_count == 3
    assert result == attempt_3


@pytest.mark.slow
def test_search_prior_art_end_to_end_over_fixture_corpus():
    patents = load_fixture_patents()
    bm25_index = build_bm25_index(patents)
    embedding_collection = build_embedding_index(patents)
    patents_by_id = {p.patent_id: p for p in patents}

    disclosure = _disclosure(
        raw_text="randomly disables neurons during training to reduce overfitting",
        technical_field="neural network training regularization",
        key_elements=["randomly disabling neurons during training", "overfitting reduction"],
    )

    results = search_prior_art(disclosure, bm25_index, embedding_collection, patents_by_id)

    assert results[0].patent_id == "US10000001B2"
