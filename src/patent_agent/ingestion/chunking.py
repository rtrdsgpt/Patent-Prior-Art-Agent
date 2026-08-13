"""Claim-level chunking: split a patent's raw claims-section text into individual `Claim`
objects, and turn those into retrieval-ready chunks.

Google Patents Public Data's `claims_localized` field is one text blob per patent (the
full claims section as filed), not pre-split into individual claims — so this is a real
parsing step, not just a formality. Claim-level is the chunk boundary used throughout the
pipeline (comparison agent output, citation-verification guard) because it's the smallest
unit that's independently legally meaningful: a claim is what gets granted, infringed, or
invalidated, not an arbitrary token window.
"""

from __future__ import annotations

import re

from patent_agent.schema import Claim

# Matches a claim boundary: a line starting with "N. " (claim number + period + whitespace).
# Anchored to line start so mid-claim references like "the method of claim 1" don't match —
# those aren't at the start of a line.
_CLAIM_BOUNDARY = re.compile(r"(?:^|\n)\s*(\d+)\s*\.\s+")

# Matches a dependency reference within a claim's own text, e.g. "of claim 1",
# "according to claims 2-4", "as recited in claim 3".
_DEPENDENCY_REF = re.compile(r"\bclaims?\s+(\d+)\b", re.IGNORECASE)


def split_claims(raw_claims_text: str) -> list[Claim]:
    """Parse a patent's raw claims-section text into individual `Claim` objects.

    Claim numbers are required to be sequential (1, 2, 3, ...) — this is what lets the
    boundary regex ignore stray "N. " patterns that aren't real claim starts (e.g. a
    claim's body text happening to contain "3. " as a sub-list marker).
    """
    matches = list(_CLAIM_BOUNDARY.finditer(raw_claims_text))
    claims: list[Claim] = []
    expected_next = 1

    for i, match in enumerate(matches):
        claim_number = int(match.group(1))
        if claim_number != expected_next:
            continue

        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw_claims_text)
        text = raw_claims_text[start:end].strip()
        if not text:
            continue

        dep_match = _DEPENDENCY_REF.search(text)
        claims.append(
            Claim(
                claim_number=claim_number,
                text=text,
                is_independent=dep_match is None,
                depends_on=int(dep_match.group(1)) if dep_match else None,
            )
        )
        expected_next += 1

    return claims


def claim_to_index_chunk(claim: Claim, patent_title: str) -> str:
    """Render a claim as a retrieval-ready chunk string.

    Prefixing with the patent title gives the embedding model context a bare legal claim
    often lacks on its own (claims frequently omit the invention's name/domain entirely,
    relying on antecedent basis from earlier claims or the spec).
    """
    return f"{patent_title}\nClaim {claim.claim_number}: {claim.text}"
