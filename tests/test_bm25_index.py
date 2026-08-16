from patent_agent.ingestion.fixtures import load_fixture_patents
from patent_agent.retrieval.bm25_index import build_bm25_index, bm25_search

PATENTS = load_fixture_patents()


def test_bm25_search_ranks_exact_terminology_match_first():
    index = build_bm25_index(PATENTS)
    results = bm25_search(index, "long short-term memory LSTM cell state", top_k=8)
    assert results[0].patent_id == "US10000003B2"


def test_bm25_search_deduplicates_to_one_result_per_patent():
    index = build_bm25_index(PATENTS)
    results = bm25_search(index, "neural network training", top_k=8)
    assert len(results) == len({r.patent_id for r in results})


def test_bm25_search_scores_are_descending():
    index = build_bm25_index(PATENTS)
    results = bm25_search(index, "convolutional neural network dropout batch normalization", top_k=8)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_bm25_search_sets_retrieval_method():
    index = build_bm25_index(PATENTS)
    results = bm25_search(index, "backpropagation", top_k=1)
    assert results[0].retrieval_method == "bm25"


def test_bm25_search_excludes_zero_score_patents():
    index = build_bm25_index(PATENTS)
    results = bm25_search(index, "zzyzx qwibble florp vandelay", top_k=8)
    assert results == []


def test_bm25_search_respects_top_k():
    index = build_bm25_index(PATENTS)
    results = bm25_search(index, "neural network", top_k=2)
    assert len(results) <= 2


def test_bm25_search_empty_index_returns_empty_list():
    index = build_bm25_index([])
    assert bm25_search(index, "neural network", top_k=5) == []
