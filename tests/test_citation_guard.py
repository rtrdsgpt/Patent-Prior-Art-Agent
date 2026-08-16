import pytest

from agents.citation_guard import verify_citations
from schema import Claim, ClaimElementComparison, NoveltyAssessment, Patent

PATENT = Patent(
    patent_id="US1234567A1",
    title="Test patent",
    abstract="An abstract.",
    claims=[
        Claim(claim_number=1, text="A method comprising: doing X;\nand doing Y.", is_independent=True),
        Claim(claim_number=2, text="The method of claim 1, further comprising doing Z.", is_independent=False, depends_on=1),
    ],
    cpc_codes=["G06N3/08"],
    assignees=["Test Corp"],
)


def _comparison(**overrides) -> ClaimElementComparison:
    fields = dict(
        disclosure_element="doing X",
        candidate_patent_id="US1234567A1",
        candidate_claim_number=1,
        cited_claim_text="doing X",
        overlap_explanation="matches",
        overlap_assessed=True,
    )
    fields.update(overrides)
    return ClaimElementComparison(**fields)


def test_verify_citations_true_when_all_citations_genuine():
    assessment = NoveltyAssessment(candidate_patent_id="US1234567A1", element_comparisons=[_comparison()])
    result = verify_citations(assessment, PATENT)
    assert result.citation_verified is True


def test_verify_citations_false_for_fabricated_quote():
    assessment = NoveltyAssessment(
        candidate_patent_id="US1234567A1",
        element_comparisons=[_comparison(cited_claim_text="doing something the claim never says")],
    )
    result = verify_citations(assessment, PATENT)
    assert result.citation_verified is False


def test_verify_citations_false_for_nonexistent_claim_number():
    assessment = NoveltyAssessment(
        candidate_patent_id="US1234567A1",
        element_comparisons=[_comparison(candidate_claim_number=99)],
    )
    result = verify_citations(assessment, PATENT)
    assert result.citation_verified is False


def test_verify_citations_true_across_multiple_comparisons_all_genuine():
    assessment = NoveltyAssessment(
        candidate_patent_id="US1234567A1",
        element_comparisons=[
            _comparison(disclosure_element="a", cited_claim_text="doing X"),
            _comparison(disclosure_element="b", cited_claim_text="doing Z", candidate_claim_number=2),
        ],
    )
    result = verify_citations(assessment, PATENT)
    assert result.citation_verified is True


def test_verify_citations_false_if_any_one_comparison_fails():
    assessment = NoveltyAssessment(
        candidate_patent_id="US1234567A1",
        element_comparisons=[
            _comparison(disclosure_element="a", cited_claim_text="doing X"),  # genuine
            _comparison(disclosure_element="b", cited_claim_text="fabricated text"),  # fabricated
        ],
    )
    result = verify_citations(assessment, PATENT)
    assert result.citation_verified is False


def test_verify_citations_tolerates_whitespace_differences():
    # Claim 1's actual text has a newline between "X;" and "and doing Y." — a quote that
    # collapses that to a single space should still verify as genuine.
    assessment = NoveltyAssessment(
        candidate_patent_id="US1234567A1",
        element_comparisons=[_comparison(cited_claim_text="doing X; and doing Y.")],
    )
    result = verify_citations(assessment, PATENT)
    assert result.citation_verified is True


def test_verify_citations_vacuously_true_for_no_comparisons():
    assessment = NoveltyAssessment(candidate_patent_id="US1234567A1", element_comparisons=[])
    result = verify_citations(assessment, PATENT)
    assert result.citation_verified is True


def test_verify_citations_raises_on_patent_id_mismatch():
    assessment = NoveltyAssessment(candidate_patent_id="US-different-patent", element_comparisons=[])
    with pytest.raises(ValueError, match="US-different-patent"):
        verify_citations(assessment, PATENT)


def test_verify_citations_does_not_mutate_original_assessment():
    assessment = NoveltyAssessment(candidate_patent_id="US1234567A1", element_comparisons=[_comparison()])
    verify_citations(assessment, PATENT)
    assert assessment.citation_verified is None
