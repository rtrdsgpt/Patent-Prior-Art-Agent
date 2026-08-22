from unittest.mock import MagicMock

import pytest

from agents.risk_report_agent import generate_risk_report
from config.settings import get_settings
from schema import ClaimElementComparison, InventionDisclosure, NoveltyAssessment

DISCLOSURE = InventionDisclosure(
    raw_text="raw",
    technical_field="neural network training regularization",
    key_elements=["dropout regularization"],
    candidate_cpc_classes=["G06N3/08"],
)


def _client_returning(content: str) -> MagicMock:
    client = MagicMock()
    client.invoke.return_value = MagicMock(content=content)
    return client


def _comparison(**overrides) -> ClaimElementComparison:
    fields = dict(
        disclosure_element="dropout regularization",
        candidate_patent_id="US1234567A1",
        candidate_claim_number=1,
        cited_claim_text="dropout",
        overlap_explanation="matches",
        overlap_assessed=True,
    )
    fields.update(overrides)
    return ClaimElementComparison(**fields)


def test_generate_risk_report_returns_fto_report_with_disclosure_and_assessments():
    assessment = NoveltyAssessment(candidate_patent_id="US1234567A1", element_comparisons=[_comparison()], citation_verified=True)
    client = _client_returning("This disclosure overlaps significantly with US1234567A1.")

    report = generate_risk_report(DISCLOSURE, [assessment], client=client)

    assert report.disclosure == DISCLOSURE
    assert report.assessments == [assessment]
    assert report.summary == "This disclosure overlaps significantly with US1234567A1."


def test_generate_risk_report_handles_no_assessments():
    client = _client_returning("No candidate prior art was found; the disclosure appears novel.")

    report = generate_risk_report(DISCLOSURE, [], client=client)

    assert report.assessments == []
    user_prompt = client.invoke.call_args.kwargs["messages"][1]["content"]
    assert "No candidate prior art was assessed" in user_prompt


def test_generate_risk_report_prompt_labels_verified_assessment():
    assessment = NoveltyAssessment(candidate_patent_id="US1234567A1", element_comparisons=[_comparison()], citation_verified=True)
    client = _client_returning("summary")

    generate_risk_report(DISCLOSURE, [assessment], client=client)

    user_prompt = client.invoke.call_args.kwargs["messages"][1]["content"]
    assert "VERIFIED" in user_prompt
    assert "UNVERIFIED" not in user_prompt


def test_generate_risk_report_prompt_flags_unverified_assessment():
    assessment = NoveltyAssessment(candidate_patent_id="US1234567A1", element_comparisons=[_comparison()], citation_verified=False)
    client = _client_returning("summary")

    generate_risk_report(DISCLOSURE, [assessment], client=client)

    user_prompt = client.invoke.call_args.kwargs["messages"][1]["content"]
    assert "UNVERIFIED" in user_prompt
    assert "citation verification failed" in user_prompt


def test_generate_risk_report_prompt_omits_overlap_details_for_unverified_assessment():
    # An unverified assessment's (unverified!) overlap claims shouldn't be quoted into the
    # prompt as if they were findings -- only the fact that verification failed should be.
    assessment = NoveltyAssessment(
        candidate_patent_id="US1234567A1",
        element_comparisons=[_comparison(overlap_explanation="a very specific unverified claim detail")],
        citation_verified=False,
    )
    client = _client_returning("summary")

    generate_risk_report(DISCLOSURE, [assessment], client=client)

    user_prompt = client.invoke.call_args.kwargs["messages"][1]["content"]
    assert "a very specific unverified claim detail" not in user_prompt


def test_generate_risk_report_no_overlap_found_is_still_labeled_verified():
    assessment = NoveltyAssessment(
        candidate_patent_id="US1234567A1",
        element_comparisons=[_comparison(overlap_assessed=False)],
        citation_verified=True,
    )
    client = _client_returning("summary")

    generate_risk_report(DISCLOSURE, [assessment], client=client)

    user_prompt = client.invoke.call_args.kwargs["messages"][1]["content"]
    assert "no substantial overlap found" in user_prompt


@pytest.mark.integration
def test_generate_risk_report_live_groq_call():
    settings = get_settings()
    if not settings.groq_api_keys:
        pytest.skip("GROQ_API_KEY not configured — set up .env to run this against live Groq")

    verified = NoveltyAssessment(candidate_patent_id="US1234567A1", element_comparisons=[_comparison()], citation_verified=True)
    unverified = NoveltyAssessment(
        candidate_patent_id="US7654321A1",
        element_comparisons=[_comparison(candidate_patent_id="US7654321A1", overlap_explanation="a hallucinated finding")],
        citation_verified=False,
    )

    report = generate_risk_report(DISCLOSURE, [verified, unverified], settings=settings)

    # openai/gpt-oss-120b (unlike the previous default model) sometimes writes patent
    # numbers with narrow no-break spaces between the letters/digits (e.g. "US 1234567
    # A1") -- cosmetic prose formatting only, the actual grounded data lives in the
    # structured assessments, not this summary (see the module's own docstring on why the
    # summary isn't authoritative). Normalize whitespace before checking rather than fight
    # the model's formatting of a free-text field.
    normalized_summary = " ".join(report.summary.split())
    assert "US 1234567 A1" in normalized_summary or "US1234567A1" in normalized_summary
    assert "hallucinated finding" not in report.summary
