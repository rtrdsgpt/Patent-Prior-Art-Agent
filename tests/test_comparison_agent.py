from unittest.mock import MagicMock

import pytest

from agents.comparison_agent import _Comparison, _ComparisonsExtraction, assess_novelty
from config.settings import get_settings
from ingestion.fixtures import load_fixture_patents
from schema import Claim, InventionDisclosure, Patent


def _client_returning(*results) -> MagicMock:
    client = MagicMock()
    client.with_structured_output.return_value.invoke.side_effect = list(results)
    return client


PATENT = Patent(
    patent_id="US1234567A1",
    title="Test patent",
    abstract="An abstract.",
    claims=[Claim(claim_number=1, text="A method comprising: doing X; and doing Y.", is_independent=True)],
    cpc_codes=["G06N3/08"],
    assignees=["Test Corp"],
)

DISCLOSURE = InventionDisclosure(
    raw_text="raw",
    technical_field="test field",
    key_elements=["doing X", "doing something unrelated"],
    candidate_cpc_classes=["G06N3/08"],
)

VALID_EXTRACTION = _ComparisonsExtraction(
    comparisons=[
        _Comparison(disclosure_element="doing X", candidate_claim_number=1, cited_claim_text="doing X", overlap_explanation="Directly matches.", overlap_assessed=True),
        _Comparison(disclosure_element="doing something unrelated", candidate_claim_number=1, cited_claim_text="doing Y", overlap_explanation="No real overlap.", overlap_assessed=False),
    ]
)


def test_assess_novelty_returns_empty_when_no_disclosure_elements():
    empty_disclosure = InventionDisclosure(raw_text="x", technical_field="x", key_elements=[], candidate_cpc_classes=[])
    client = MagicMock()

    result = assess_novelty(empty_disclosure, PATENT, client=client)

    assert result.candidate_patent_id == "US1234567A1"
    assert result.element_comparisons == []
    client.with_structured_output.assert_not_called()


def test_assess_novelty_returns_empty_when_patent_has_no_independent_claims():
    no_independent = Patent(
        patent_id="US9999999A1",
        title="x",
        abstract="x",
        claims=[Claim(claim_number=1, text="x", is_independent=False, depends_on=None)],
        cpc_codes=[],
        assignees=[],
    )
    client = MagicMock()

    result = assess_novelty(DISCLOSURE, no_independent, client=client)

    assert result.element_comparisons == []
    client.with_structured_output.assert_not_called()


def test_assess_novelty_builds_comparisons_with_correct_patent_id():
    client = _client_returning(VALID_EXTRACTION)

    result = assess_novelty(DISCLOSURE, PATENT, client=client)

    assert result.candidate_patent_id == "US1234567A1"
    assert len(result.element_comparisons) == 2
    assert all(c.candidate_patent_id == "US1234567A1" for c in result.element_comparisons)


def test_assess_novelty_preserves_overlap_assessed_flags():
    client = _client_returning(VALID_EXTRACTION)

    result = assess_novelty(DISCLOSURE, PATENT, client=client)

    by_element = {c.disclosure_element: c for c in result.element_comparisons}
    assert by_element["doing X"].overlap_assessed is True
    assert by_element["doing something unrelated"].overlap_assessed is False


def test_assess_novelty_retries_on_invalid_claim_number():
    invalid_claim_number = _ComparisonsExtraction(
        comparisons=[
            _Comparison(disclosure_element="doing X", candidate_claim_number=99, cited_claim_text="doing X", overlap_explanation="x", overlap_assessed=True),  # doesn't exist on PATENT
            _Comparison(disclosure_element="doing something unrelated", candidate_claim_number=1, cited_claim_text="doing Y", overlap_explanation="x", overlap_assessed=False),
        ]
    )
    client = _client_returning(invalid_claim_number, VALID_EXTRACTION)

    result = assess_novelty(DISCLOSURE, PATENT, client=client)

    assert client.with_structured_output.return_value.invoke.call_count == 2
    assert len(result.element_comparisons) == 2


def test_assess_novelty_retries_when_a_disclosure_element_is_missing():
    incomplete = _ComparisonsExtraction(
        comparisons=[_Comparison(disclosure_element="doing X", candidate_claim_number=1, cited_claim_text="doing X", overlap_explanation="x", overlap_assessed=True)]
    )
    client = _client_returning(incomplete, VALID_EXTRACTION)

    result = assess_novelty(DISCLOSURE, PATENT, client=client)

    assert client.with_structured_output.return_value.invoke.call_count == 2
    assert len(result.element_comparisons) == 2


def test_assess_novelty_includes_claim_elements_as_reference_context():
    client = _client_returning(VALID_EXTRACTION)

    assess_novelty(DISCLOSURE, PATENT, claim_elements={1: ["doing X", "doing Y"]}, client=client)

    user_prompt = client.with_structured_output.return_value.invoke.call_args.args[0][1]["content"]
    assert "pre-structured breakdown" in user_prompt
    assert "doing X" in user_prompt


def test_assess_novelty_omits_reference_context_when_claim_elements_not_given():
    client = _client_returning(VALID_EXTRACTION)

    assess_novelty(DISCLOSURE, PATENT, client=client)

    user_prompt = client.with_structured_output.return_value.invoke.call_args.args[0][1]["content"]
    assert "pre-structured breakdown" not in user_prompt


def test_assess_novelty_citation_verified_left_unset():
    client = _client_returning(VALID_EXTRACTION)
    result = assess_novelty(DISCLOSURE, PATENT, client=client)
    assert result.citation_verified is None


@pytest.mark.integration
def test_assess_novelty_live_groq_call():
    settings = get_settings()
    if not settings.groq_api_keys:
        pytest.skip("GROQ_API_KEY not configured — set up .env to run this against live Groq")

    dropout_patent = next(p for p in load_fixture_patents() if p.patent_id == "US10000001B2")
    disclosure = InventionDisclosure(
        raw_text="raw",
        technical_field="neural network training regularization",
        key_elements=["randomly deactivating neurons during training", "unrelated blockchain consensus mechanism"],
        candidate_cpc_classes=["G06N3/08"],
    )

    result = assess_novelty(disclosure, dropout_patent, settings=settings)

    by_element = {c.disclosure_element: c for c in result.element_comparisons}
    assert by_element["randomly deactivating neurons during training"].overlap_assessed is True
    assert by_element["unrelated blockchain consensus mechanism"].overlap_assessed is False
    # cited_claim_text should be real quotes, checkable against the actual claim text
    for comparison in result.element_comparisons:
        claim_text = next(c.text for c in dropout_patent.claims if c.claim_number == comparison.candidate_claim_number)
        assert comparison.cited_claim_text in claim_text
