from unittest.mock import MagicMock

import pytest

from ingestion.fixtures import load_fixture_patents
from retrieval.embedding_index import build_embedding_index, dense_search

pytestmark = pytest.mark.slow  # loads a real local embedding model


@pytest.fixture(scope="module")
def collection():
    return build_embedding_index(load_fixture_patents())


def test_dense_search_returns_requested_top_k(collection):
    results = dense_search(collection, "training a neural network", top_k=3)
    assert len(results) == 3


def test_dense_search_ranks_semantically_closest_patent_first(collection):
    results = dense_search(collection, "randomly dropping neurons during training to prevent overfitting", top_k=8)
    assert results[0].patent_id == "US10000001B2"


def test_dense_search_deduplicates_to_one_result_per_patent(collection):
    results = dense_search(collection, "neural network", top_k=8)
    assert len(results) == len({r.patent_id for r in results})


def test_dense_search_scores_are_descending(collection):
    results = dense_search(collection, "neural network training", top_k=8)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_dense_search_sets_retrieval_method(collection):
    results = dense_search(collection, "neural network", top_k=1)
    assert results[0].retrieval_method == "dense"


def test_build_embedding_index_batches_upsert_beyond_max_batch_size():
    # A real Chroma client rejects a single upsert() beyond its own max batch size --
    # confirmed by actually hitting it while indexing a ~1500-patent corpus (see log.md).
    # Uses a mocked client with a small limit so this stays a fast check, not a
    # multi-thousand-document indexing run.
    mock_collection = MagicMock()
    mock_client = MagicMock()
    mock_client.get_max_batch_size.return_value = 3
    mock_client.get_or_create_collection.return_value = mock_collection

    patents = load_fixture_patents()  # 8 patents, well over 3 claim-chunks combined
    build_embedding_index(patents, client=mock_client)

    assert mock_collection.upsert.call_count > 1
    for call in mock_collection.upsert.call_args_list:
        assert len(call.kwargs["ids"]) <= 3
