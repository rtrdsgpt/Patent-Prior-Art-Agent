from unittest.mock import MagicMock

import pytest

from agents.orchestrator import run_fto_pipeline
from config.settings import Settings, get_settings
from ingestion.fixtures import load_fixture_patents
from schema import FTOReport, InventionDisclosure, NoveltyAssessment, SearchResult

PATENTS = load_fixture_patents()
PATENTS_BY_ID = {p.patent_id: p for p in PATENTS}

DISCLOSURE = InventionDisclosure(raw_text="raw", technical_field="x", key_elements=["dropout"], candidate_cpc_classes=["G06N3/08"])


def _patch_pipeline(monkeypatch, *, candidates=None, assessment=None, report=None):
    parse_disclosure_mock = MagicMock(return_value=DISCLOSURE)
    search_mock = MagicMock(return_value=candidates if candidates is not None else [SearchResult(patent_id="US10000001B2", score=1.0, retrieval_method="reranked")])
    claim_elements_mock = MagicMock(return_value={1: ["doing X"]})
    assess_novelty_mock = MagicMock(return_value=assessment or NoveltyAssessment(candidate_patent_id="US10000001B2", element_comparisons=[]))
    verify_mock = MagicMock(side_effect=lambda a, p: a.model_copy(update={"citation_verified": True}))
    risk_report_mock = MagicMock(return_value=report or FTOReport(disclosure=DISCLOSURE, assessments=[], summary="summary"))

    monkeypatch.setattr("agents.orchestrator.parse_disclosure", parse_disclosure_mock)
    monkeypatch.setattr("agents.orchestrator.search_prior_art", search_mock)
    monkeypatch.setattr("agents.orchestrator.parse_claim_elements", claim_elements_mock)
    monkeypatch.setattr("agents.orchestrator.assess_novelty", assess_novelty_mock)
    monkeypatch.setattr("agents.orchestrator.verify_citations", verify_mock)
    monkeypatch.setattr("agents.orchestrator.generate_risk_report", risk_report_mock)

    return dict(
        parse_disclosure=parse_disclosure_mock,
        search=search_mock,
        claim_elements=claim_elements_mock,
        assess_novelty=assess_novelty_mock,
        verify=verify_mock,
        risk_report=risk_report_mock,
    )


def test_run_fto_pipeline_calls_stages_in_order(monkeypatch):
    mocks = _patch_pipeline(monkeypatch)

    result = run_fto_pipeline("some disclosure text", bm25_index=MagicMock(), embedding_collection=MagicMock(), patents_by_id=PATENTS_BY_ID, client=MagicMock())

    mocks["parse_disclosure"].assert_called_once()
    mocks["search"].assert_called_once()
    mocks["claim_elements"].assert_called_once()
    mocks["assess_novelty"].assert_called_once()
    mocks["verify"].assert_called_once()
    mocks["risk_report"].assert_called_once()
    assert isinstance(result, FTOReport)


def test_run_fto_pipeline_passes_claim_elements_into_comparison(monkeypatch):
    mocks = _patch_pipeline(monkeypatch)

    run_fto_pipeline("x", bm25_index=MagicMock(), embedding_collection=MagicMock(), patents_by_id=PATENTS_BY_ID, client=MagicMock())

    call_kwargs = mocks["assess_novelty"].call_args.kwargs
    assert call_kwargs["claim_elements"] == {1: ["doing X"]}


def test_run_fto_pipeline_verifies_citations_before_risk_report(monkeypatch):
    unverified = NoveltyAssessment(candidate_patent_id="US10000001B2", element_comparisons=[])
    mocks = _patch_pipeline(monkeypatch, assessment=unverified)

    run_fto_pipeline("x", bm25_index=MagicMock(), embedding_collection=MagicMock(), patents_by_id=PATENTS_BY_ID, client=MagicMock())

    # generate_risk_report should receive assessments that went through verify_citations,
    # i.e. citation_verified is set (True, per the mock), not the raw unverified assessment.
    assessments_passed = mocks["risk_report"].call_args.args[1]
    assert assessments_passed[0].citation_verified is True


def test_run_fto_pipeline_handles_zero_candidates(monkeypatch):
    mocks = _patch_pipeline(monkeypatch, candidates=[])

    run_fto_pipeline("x", bm25_index=MagicMock(), embedding_collection=MagicMock(), patents_by_id=PATENTS_BY_ID, client=MagicMock())

    mocks["assess_novelty"].assert_not_called()
    assert mocks["risk_report"].call_args.args == (DISCLOSURE, [])


def test_run_fto_pipeline_skips_candidate_missing_from_patents_by_id(monkeypatch):
    mocks = _patch_pipeline(monkeypatch, candidates=[SearchResult(patent_id="does-not-exist", score=1.0, retrieval_method="reranked")])

    run_fto_pipeline("x", bm25_index=MagicMock(), embedding_collection=MagicMock(), patents_by_id=PATENTS_BY_ID, client=MagicMock())

    mocks["assess_novelty"].assert_not_called()


def test_run_fto_pipeline_fans_out_and_accumulates_assessments_for_multiple_candidates(monkeypatch):
    # The Send-based fan-out is the actual "why LangGraph now" claim in orchestrator.py's
    # docstring -- worth a dedicated test with >1 candidate, not just the 1-candidate case
    # every other test in this file already exercises.
    candidates = [
        SearchResult(patent_id="US10000001B2", score=3.0, retrieval_method="reranked"),
        SearchResult(patent_id="US10000002B2", score=2.0, retrieval_method="reranked"),
        SearchResult(patent_id="US10000003B2", score=1.0, retrieval_method="reranked"),
    ]

    def fake_assess_novelty(disclosure, patent, claim_elements=None, client=None, settings=None):
        return NoveltyAssessment(candidate_patent_id=patent.patent_id, element_comparisons=[])

    mocks = _patch_pipeline(monkeypatch, candidates=candidates)
    mocks["assess_novelty"].side_effect = fake_assess_novelty

    run_fto_pipeline("x", bm25_index=MagicMock(), embedding_collection=MagicMock(), patents_by_id=PATENTS_BY_ID, client=MagicMock())

    assert mocks["assess_novelty"].call_count == 3
    assessments_passed = mocks["risk_report"].call_args.args[1]
    assert {a.candidate_patent_id for a in assessments_passed} == {c.patent_id for c in candidates}


def test_run_fto_pipeline_reuses_one_client_across_all_stages(monkeypatch):
    mocks = _patch_pipeline(monkeypatch)
    shared_client = MagicMock()

    run_fto_pipeline("x", bm25_index=MagicMock(), embedding_collection=MagicMock(), patents_by_id=PATENTS_BY_ID, client=shared_client)

    assert mocks["parse_disclosure"].call_args.kwargs["client"] is shared_client
    assert mocks["claim_elements"].call_args.kwargs["client"] is shared_client
    assert mocks["assess_novelty"].call_args.kwargs["client"] is shared_client
    assert mocks["risk_report"].call_args.kwargs["client"] is shared_client


@pytest.mark.integration
def test_run_fto_pipeline_live_end_to_end():
    settings = get_settings()
    if not settings.groq_api_keys:
        pytest.skip("GROQ_API_KEY not configured — set up .env to run this against live Groq")

    from retrieval.bm25_index import build_bm25_index
    from retrieval.embedding_index import build_embedding_index

    bm25_index = build_bm25_index(PATENTS)
    embedding_collection = build_embedding_index(PATENTS)
    # Small rerank_top_k to bound how many live Groq calls this test makes (2 per candidate
    # -- claims-parser + comparison -- plus 1 disclosure-parser + 1 risk-report).
    bounded_settings = Settings(groq_api_key=settings.groq_api_key, rerank_top_k=2)

    report = run_fto_pipeline(
        "A neural network training method that randomly disables neurons during training "
        "to prevent overfitting, using dropout regularization.",
        bm25_index,
        embedding_collection,
        PATENTS_BY_ID,
        settings=bounded_settings,
    )

    assert isinstance(report, FTOReport)
    assert len(report.assessments) == 2
    assert all(a.citation_verified is not None for a in report.assessments)
    assert report.summary
