"""Citation-verification guard (todo.md section 2): deterministic, no LLM involved.

Re-checks that every `cited_claim_text` a `NoveltyAssessment`'s comparisons quote actually
appears in the claim it claims to quote from. This is what makes the pipeline's grounding
claim real rather than aspirational — `comparison_agent.py`'s prompt *asks* the model to
quote verbatim, but asking isn't verifying; this is the check that catches it when the model
doesn't.

The actual text-match logic delegates to the shared `grounded_evals.verify_citation` — this
project's own README already commits to sharing retrieval/citation-verification metrics
across the CV portfolio via that package (see its own docstring: "the reusable form of the
hallucination-guard pattern"), so hand-rolling a second implementation here would have been
duplicating exactly the thing that package exists to share. `grounded_evals.verify_citation`
also does more than the whitespace-only normalization this module's first version had:
case-insensitive comparison, plus a fuzzy-match fallback (difflib ratio over a sliding
window) for minor punctuation/quoting drift beyond just whitespace — for free, by reusing it
instead of maintaining a narrower local version.
"""

from __future__ import annotations

from grounded_evals import verify_citation as _lexical_citation_match

from schema import NoveltyAssessment, Patent


def _claim_text(patent: Patent, claim_number: int) -> str | None:
    return next((c.text for c in patent.claims if c.claim_number == claim_number), None)


def verify_citations(assessment: NoveltyAssessment, patent: Patent) -> NoveltyAssessment:
    """Return a copy of `assessment` with `citation_verified` set: `True` only if every
    comparison's `cited_claim_text` genuinely matches (exactly or near-exactly — see
    `grounded_evals.verify_citation`) the claim it names; `False` if any comparison fails,
    including citing a claim number that doesn't exist on `patent` at all. Vacuously `True`
    for an assessment with no comparisons — nothing to fail.
    """
    if assessment.candidate_patent_id != patent.patent_id:
        raise ValueError(f"Assessment is for patent {assessment.candidate_patent_id!r}, not {patent.patent_id!r}")

    all_verified = True
    for comparison in assessment.element_comparisons:
        claim_text = _claim_text(patent, comparison.candidate_claim_number)
        if claim_text is None or not _lexical_citation_match(comparison.cited_claim_text, claim_text):
            all_verified = False
            break

    return assessment.model_copy(update={"citation_verified": all_verified})
