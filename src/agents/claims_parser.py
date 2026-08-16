"""Claims-parser agent (todo.md section 2): structure a candidate patent's independent
claims into discrete elements — conceptually similar to Legal SLM SFT's IRAC structuring,
applied to patent claim language instead.

One Groq call per patent (all independent claims batched together, not one call per claim)
to keep LLM call count at O(candidates) rather than O(candidates × claims) — this runs once
per search-result candidate, so that multiplier matters for latency/cost.

The elements this produces are a *reasoning aid* for the comparison agent, not themselves
the source of truth for citation — they're meant to track the claim's own language closely,
but the comparison agent still quotes directly from `Claim.text`, and the citation-
verification guard (`agents/citation_guard.py`) checks against that raw text, not against
anything this module outputs. So a claims-parser element that paraphrases slightly doesn't
compromise correctness downstream, only reasoning clarity.
"""

from __future__ import annotations

from agents.groq_client import RotatingGroqClient, build_groq_client
from agents.groq_json import request_json
from config.settings import Settings, get_settings
from schema import Patent
from tracing import traced

_SYSTEM_PROMPT = """You are a patent claims analyst. You will be given one or more independent \
patent claims. For each claim, break it into its discrete elements — the individual technical \
steps, components, or requirements the claim recites (typically claim drafting separates these \
with semicolons or "and"). Stay as close to the claim's own wording as possible; don't add, \
omit, or rephrase substantive content. Respond with ONLY a JSON object of this shape:

{"claims": [{"claim_number": <int>, "elements": [<string>, ...]}, ...]}

Include one entry per claim you were given, in any order."""


def _build_user_prompt(independent_claims) -> str:
    claims_text = "\n\n".join(f"Claim {c.claim_number}: {c.text}" for c in independent_claims)
    return f"Independent claims:\n\n{claims_text}"


def _validate(parsed: dict, expected_claim_numbers: set[int]) -> dict[int, list[str]]:
    result = {entry["claim_number"]: list(entry["elements"]) for entry in parsed["claims"]}
    missing = expected_claim_numbers - result.keys()
    if missing:
        raise ValueError(f"Response is missing elements for claim number(s): {sorted(missing)}")
    return result


@traced("claims_parser")
def parse_claim_elements(
    patent: Patent,
    client: RotatingGroqClient | None = None,
    settings: Settings | None = None,
) -> dict[int, list[str]]:
    """Return `{claim_number: [element, ...]}` for every independent claim of `patent`.

    Patents with no independent claims (shouldn't happen given `chunking.split_claims`
    always marks the first parsed claim independent, but not assumed) return `{}` without
    calling the LLM at all — nothing to structure.
    """
    independent_claims = patent.independent_claims
    if not independent_claims:
        return {}

    settings = settings or get_settings()
    client = client or build_groq_client(settings)
    expected = {c.claim_number for c in independent_claims}

    return request_json(
        client,
        settings,
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=_build_user_prompt(independent_claims),
        validate=lambda parsed: _validate(parsed, expected),
    )
