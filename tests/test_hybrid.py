import pytest

from config.settings import Settings
from ingestion.fixtures import load_fixture_patents
from retrieval.bm25_index import build_bm25_index
from retrieval.embedding_index import build_embedding_index
from retrieval.hybrid import hybrid_search, reciprocal_rank_fusion
from schema import SearchResult


def _results(*patent_ids: str, method: str = "bm25") -> list[SearchResult]:
    return [
        SearchResult(patent_id=pid, score=float(len(patent_ids) - i), retrieval_method=method)
        for i, pid in enumerate(patent_ids)
    ]


def test_rrf_ranks_patent_agreed_on_by_both_rankers_first():
    bm25 = _results("A", "B", "C", method="bm25")
    dense = _results("C", "A", "B", method="dense")
    fused = reciprocal_rank_fusion([bm25, dense], top_k=3)
    assert fused[0].patent_id == "A"


def test_rrf_includes_patent_found_by_only_one_ranker():
    bm25 = _results("A", "B", method="bm25")
    dense = _results("C", "D", method="dense")
    fused = reciprocal_rank_fusion([bm25, dense], top_k=10)
    assert {r.patent_id for r in fused} == {"A", "B", "C", "D"}


def test_rrf_tags_retrieval_method_as_hybrid():
    fused = reciprocal_rank_fusion([_results("A")], top_k=1)
    assert fused[0].retrieval_method == "hybrid"


def test_rrf_respects_top_k():
    fused = reciprocal_rank_fusion([_results("A", "B", "C")], top_k=2)
    assert len(fused) == 2


def test_rrf_scores_are_descending():
    bm25 = _results("A", "B", "C", method="bm25")
    dense = _results("B", "C", "A", method="dense")
    fused = reciprocal_rank_fusion([bm25, dense], top_k=3)
    scores = [r.score for r in fused]
    assert scores == sorted(scores, reverse=True)


def test_rrf_empty_lists_returns_empty():
    assert reciprocal_rank_fusion([[], []], top_k=5) == []


@pytest.mark.slow
def test_hybrid_search_surfaces_lexical_and_semantic_matches():
    patents = load_fixture_patents()
    bm25_index = build_bm25_index(patents)
    embedding_collection = build_embedding_index(patents)
    settings = Settings(bm25_top_k=10, dense_top_k=10, hybrid_top_k=5)

    results = hybrid_search(bm25_index, embedding_collection, "dropout regularization during neural network training", settings=settings)

    assert results[0].patent_id == "US10000001B2"
    assert all(r.retrieval_method == "hybrid" for r in results)
    assert len(results) <= 5
