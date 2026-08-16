"""Comparison/novelty-assessment agent (todo.md section 2): element-by-element, RAG-grounded
overlap assessment between a disclosure's key elements and a candidate patent's independent
claims — every comparison citing specific claim text.

One Groq call per candidate patent (all independent claims and all disclosure elements in a
single request), same O(candidates) batching rationale as `claims_parser.py`. The model is
asked to quote claim text verbatim, but "asked to" isn't "guaranteed to" — `cited_claim_text`
is NOT trusted as accurate here. `agents/citation_guard.py` deterministically re-checks every
quote against the real `Claim.text` before anything downstream treats it as grounded; this
agent's job is producing the comparison, not verifying it.
"""

from __future__ import annotations

from agents.groq_client import RotatingGroqClient, build_groq_client
from agents.groq_json import request_json
from config.settings import Settings, get_settings
from schema import ClaimElementComparison, InventionDisclosure, NoveltyAssessment, Patent

_SYSTEM_PROMPT = """You are a patent comparison analyst performing a freedom-to-operate \
element-by-element overlap assessment. You will be given a list of disclosure elements and a \
candidate patent's independent claims. For EACH disclosure element, determine whether any of the \
candidate's independent claims covers ("reads on") it:

- Pick the single most relevant claim to compare it against (even if there's no real overlap —
  pick whichever claim is topically closest).
- Quote a short, EXACT, contiguous substring from that claim's own text as "cited_claim_text" —
  copy it verbatim, do not paraphrase or combine text from different parts of the claim.
- Explain the overlap (or lack of it) in "overlap_explanation".
- Set "overlap_assessed" to true only if the claim substantially covers the disclosure element,
  false otherwise.
- Echo the disclosure element back EXACTLY as given in "disclosure_element" (don't reword it).

Respond with ONLY a JSON object of this shape:

{"comparisons": [{"disclosure_element": <string>, "candidate_claim_number": <int>, \
"cited_claim_text": <string>, "overlap_explanation": <string>, "overlap_assessed": <bool>}, ...]}

Include exactly one entry per disclosure element you were given."""


def _build_user_prompt(disclosure: InventionDisclosure, independent_claims) -> str:
    elements_text = "\n".join(f"- {e}" for e in disclosure.key_elements)
    claims_text = "\n\n".join(f"Claim {c.claim_number}: {c.text}" for c in independent_claims)
    return f"Disclosure elements:\n{elements_text}\n\nCandidate patent's independent claims:\n\n{claims_text}"


def _validate(parsed: dict, patent_id: str, disclosure_elements: list[str], valid_claim_numbers: set[int]) -> list[ClaimElementComparison]:
    comparisons = []
    seen_elements = set()

    for entry in parsed["comparisons"]:
        claim_number = entry["candidate_claim_number"]
        if claim_number not in valid_claim_numbers:
            raise ValueError(f"candidate_claim_number {claim_number} is not one of this patent's independent claims {sorted(valid_claim_numbers)}")

        comparisons.append(
            ClaimElementComparison(
                disclosure_element=entry["disclosure_element"],
                candidate_patent_id=patent_id,
                candidate_claim_number=claim_number,
                cited_claim_text=entry["cited_claim_text"],
                overlap_explanation=entry["overlap_explanation"],
                overlap_assessed=entry["overlap_assessed"],
            )
        )
        seen_elements.add(entry["disclosure_element"])

    missing = set(disclosure_elements) - seen_elements
    if missing:
        raise ValueError(f"Response is missing comparisons for disclosure element(s): {sorted(missing)}")

    return comparisons


def assess_novelty(
    disclosure: InventionDisclosure,
    patent: Patent,
    client: RotatingGroqClient | None = None,
    settings: Settings | None = None,
) -> NoveltyAssessment:
    """Compare every element of `disclosure` against `patent`'s independent claims.

    Returns an assessment with no comparisons (not an error) if either side has nothing to
    compare — no disclosure elements extracted, or the patent has no independent claims.
    `citation_verified` is left unset here; that's `citation_guard.py`'s job, run afterward.
    """
    independent_claims = patent.independent_claims
    if not independent_claims or not disclosure.key_elements:
        return NoveltyAssessment(candidate_patent_id=patent.patent_id, element_comparisons=[])

    settings = settings or get_settings()
    client = client or build_groq_client(settings)
    valid_claim_numbers = {c.claim_number for c in independent_claims}

    comparisons = request_json(
        client,
        settings,
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=_build_user_prompt(disclosure, independent_claims),
        validate=lambda parsed: _validate(parsed, patent.patent_id, disclosure.key_elements, valid_claim_numbers),
    )

    return NoveltyAssessment(candidate_patent_id=patent.patent_id, element_comparisons=comparisons)
