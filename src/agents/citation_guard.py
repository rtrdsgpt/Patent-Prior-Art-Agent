"""Citation-verification guard (todo.md section 2): deterministic, no LLM involved.

Re-checks that every `cited_claim_text` a `NoveltyAssessment`'s comparisons quote actually
appears in the claim it claims to quote from. This is what makes the pipeline's grounding
claim real rather than aspirational — `comparison_agent.py`'s prompt *asks* the model to
quote verbatim, but asking isn't verifying; this is the check that catches it when the model
doesn't. Same "reuse the hallucination-guard pattern from Exporter Crawl's rank_engine.py"
role todo.md calls for, just against this project's own data shapes.
"""

from __future__ import annotations

import re

from schema import NoveltyAssessment, Patent


def _normalize(text: str) -> str:
    """Collapse whitespace so line breaks/extra spaces the model introduces when quoting
    don't fail an otherwise-genuine citation — this stays a strict substring check, just not
    brittle to formatting-only differences."""
    return re.sub(r"\s+", " ", text).strip()


def _claim_text(patent: Patent, claim_number: int) -> str | None:
    return next((c.text for c in patent.claims if c.claim_number == claim_number), None)


def verify_citations(assessment: NoveltyAssessment, patent: Patent) -> NoveltyAssessment:
    """Return a copy of `assessment` with `citation_verified` set: `True` only if every
    comparison's `cited_claim_text` is a genuine (whitespace-normalized) substring of the
    claim it names; `False` if any comparison fails, including citing a claim number that
    doesn't exist on `patent` at all. Vacuously `True` for an assessment with no comparisons
    — nothing to fail.
    """
    if assessment.candidate_patent_id != patent.patent_id:
        raise ValueError(f"Assessment is for patent {assessment.candidate_patent_id!r}, not {patent.patent_id!r}")

    all_verified = True
    for comparison in assessment.element_comparisons:
        claim_text = _claim_text(patent, comparison.candidate_claim_number)
        if claim_text is None or _normalize(comparison.cited_claim_text) not in _normalize(claim_text):
            all_verified = False
            break

    return assessment.model_copy(update={"citation_verified": all_verified})
