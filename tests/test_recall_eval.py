from unittest.mock import MagicMock

import pytest

from config.settings import Settings
from evaluation.recall_eval import EvalCase, build_eval_set, run_recall_eval
from ingestion.fixtures import load_fixture_patents
from retrieval.bm25_index import build_bm25_index
from retrieval.embedding_index import build_embedding_index
from schema import SearchResult

PATENTS = load_fixture_patents()
PATENTS_BY_ID = {p.patent_id: p for p in PATENTS}


def test_build_eval_set_only_includes_patents_with_examiner_citations():
    cases = build_eval_set(PATENTS)
    case_ids = {c.patent_id for c in cases}
    # From the fixture corpus: US10000002B2, US10000004B2, US10000006B2, and US10000007B2
    # each carry an EXA citation; the rest don't (see tests/fixtures/sample_patents.json).
    assert case_ids == {"US10000002B2", "US10000004B2", "US10000006B2", "US10000007B2"}


def test_build_eval_set_query_text_combines_title_and_abstract():
    cases = build_eval_set(PATENTS)
    case = next(c for c in cases if c.patent_id == "US10000002B2")
    patent = PATENTS_BY_ID["US10000002B2"]
    assert patent.title in case.query_text
    assert patent.abstract in case.query_text


def test_build_eval_set_relevant_ids_match_examiner_citations():
    cases = build_eval_set(PATENTS)
    case = next(c for c in cases if c.patent_id == "US10000002B2")
    assert case.relevant_patent_ids == {"US10000001B2"}


def test_build_eval_set_in_corpus_relevant_ids_are_full_here_since_fixture_corpus_is_self_contained():
    cases = build_eval_set(PATENTS)
    for case in cases:
        assert case.in_corpus_relevant_patent_ids == case.relevant_patent_ids


def test_build_eval_set_excludes_citations_pointing_outside_the_corpus():
    from schema import Claim, Patent

    patents = PATENTS + [
        Patent(
            patent_id="US_EXTRA",
            title="Extra patent",
            abstract="abstract",
            claims=[Claim(claim_number=1, text="A method.", is_independent=True)],
            cpc_codes=["G06N3/08"],
            assignees=[],
            citations=[
                {"cited_patent_id": "US10000001B2", "category": "EXA"},  # in corpus
                {"cited_patent_id": "US-not-in-corpus", "category": "EXA"},  # not in corpus
            ],
        )
    ]
    cases = build_eval_set(patents)
    case = next(c for c in cases if c.patent_id == "US_EXTRA")
    assert case.relevant_patent_ids == {"US10000001B2", "US-not-in-corpus"}
    assert case.in_corpus_relevant_patent_ids == {"US10000001B2"}


def test_run_recall_eval_respects_sample_size(monkeypatch):
    monkeypatch.setattr("evaluation.recall_eval.hybrid_search", MagicMock(return_value=[]))
    monkeypatch.setattr("evaluation.recall_eval.rerank", MagicMock(return_value=[]))
    cases = build_eval_set(PATENTS)

    result = run_recall_eval(cases, bm25_index=MagicMock(), embedding_collection=MagicMock(), patents_by_id=PATENTS_BY_ID, sample_size=1)

    assert result.num_cases == 1


def test_run_recall_eval_filters_query_patent_self_match(monkeypatch):
    monkeypatch.setattr("evaluation.recall_eval.hybrid_search", MagicMock(return_value=[]))
    # Self-match ranked first, real citation second -- filtering the self-match should still
    # leave the real citation visible to recall@k.
    monkeypatch.setattr(
        "evaluation.recall_eval.rerank",
        MagicMock(return_value=[
            SearchResult(patent_id="US10000002B2", score=9.0, retrieval_method="reranked"),
            SearchResult(patent_id="US10000001B2", score=5.0, retrieval_method="reranked"),
        ]),
    )
    case = EvalCase(patent_id="US10000002B2", query_text="x", relevant_patent_ids={"US10000001B2"}, in_corpus_relevant_patent_ids={"US10000001B2"})

    result = run_recall_eval([case], bm25_index=MagicMock(), embedding_collection=MagicMock(), patents_by_id=PATENTS_BY_ID, k=5)

    assert result.overall.mean_recall_at_k == 1.0


def test_run_recall_eval_in_corpus_is_none_when_no_case_has_in_corpus_citations(monkeypatch):
    monkeypatch.setattr("evaluation.recall_eval.hybrid_search", MagicMock(return_value=[]))
    monkeypatch.setattr("evaluation.recall_eval.rerank", MagicMock(return_value=[]))
    case = EvalCase(patent_id="US10000002B2", query_text="x", relevant_patent_ids={"US-outside-corpus"}, in_corpus_relevant_patent_ids=set())

    result = run_recall_eval([case], bm25_index=MagicMock(), embedding_collection=MagicMock(), patents_by_id=PATENTS_BY_ID)

    assert result.num_in_corpus_cases == 0
    assert result.in_corpus is None


def test_run_recall_eval_passes_k_plus_one_rerank_top_k_to_make_room_for_self_match(monkeypatch):
    hybrid_mock = MagicMock(return_value=[])
    rerank_mock = MagicMock(return_value=[])
    monkeypatch.setattr("evaluation.recall_eval.hybrid_search", hybrid_mock)
    monkeypatch.setattr("evaluation.recall_eval.rerank", rerank_mock)
    case = EvalCase(patent_id="US10000002B2", query_text="x", relevant_patent_ids={"US10000001B2"}, in_corpus_relevant_patent_ids={"US10000001B2"})

    run_recall_eval([case], bm25_index=MagicMock(), embedding_collection=MagicMock(), patents_by_id=PATENTS_BY_ID, settings=Settings(), k=5)

    used_settings = hybrid_mock.call_args.kwargs["settings"]
    assert used_settings.rerank_top_k == 6


@pytest.mark.slow
def test_run_recall_eval_end_to_end_over_fixture_corpus():
    bm25_index = build_bm25_index(PATENTS)
    embedding_collection = build_embedding_index(PATENTS)
    cases = build_eval_set(PATENTS)

    result = run_recall_eval(cases, bm25_index, embedding_collection, PATENTS_BY_ID, k=5)

    assert result.num_cases == 4
    assert result.num_in_corpus_cases == 4
    # The dropout patent (US10000001B2) is lexically/semantically close to its two citing
    # patents, so it should be recoverable within top-5 for at least one of them.
    assert result.overall.mean_recall_at_k > 0.0
