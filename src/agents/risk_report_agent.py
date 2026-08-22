"""Risk-report/critic agent (todo.md section 2): aggregates citation-verified novelty
assessments into a structured `FTOReport`.

The citation-verification guard (`agents/citation_guard.py`) already ran by the time this
agent sees `assessments` — its job is aggregation and narrative, not verification. Output is
free-text prose (`FTOReport.summary`), not JSON, so this doesn't use `agents/groq_json.py`'s
retry loop — there's no schema for a summary string to fail against.

**Important limitation, stated plainly rather than glossed over:** the prompt instructs the
model to describe `citation_verified=False` assessments as needing manual review rather than
as confirmed findings, but that's an instruction the model can follow imperfectly — nothing
deterministically enforces that the prose *respects* the distinction the way
`citation_guard.py` deterministically enforces the citations themselves. The authoritative
signal for what's trustworthy is the structured `citation_verified` field on each
`NoveltyAssessment` in the report, not the summary prose — any consumer of this report (API
client, MCP tool, a human reviewer) that cares about grounding should filter on that field
directly rather than trusting the narrative alone.
"""

from __future__ import annotations

from agents.groq_client import RotatingChatGroq, build_groq_client
from config.settings import Settings, get_settings
from schema import FTOReport, InventionDisclosure, NoveltyAssessment
from tracing import traced

_SYSTEM_PROMPT = """You are drafting the summary section of a patent freedom-to-operate (FTO) \
risk report. You'll be given an invention disclosure and a list of candidate prior-art patents \
with element-by-element overlap assessments against that disclosure. Each candidate is labeled \
either VERIFIED (its quoted claim text was independently confirmed to genuinely appear in the \
patent) or UNVERIFIED (verification failed — the quoted text could not be confirmed).

Write a concise prose summary (3-6 sentences) of the overall novelty/FTO risk:
- Base your risk assessment ONLY on VERIFIED candidates' overlap_assessed=true comparisons.
- Name the highest-risk candidate patent(s) if any show substantial verified overlap.
- Explicitly flag any UNVERIFIED candidates as requiring manual review — never describe their
  findings as confirmed.
- If no VERIFIED candidate shows meaningful overlap, say the disclosure appears novel relative
  to the searched corpus, and still flag any UNVERIFIED candidates separately.

Respond with ONLY the prose summary — no headers, no JSON, no markdown."""


def _describe_assessment(assessment: NoveltyAssessment) -> str:
    if assessment.citation_verified is False:
        return f"Candidate {assessment.candidate_patent_id}: UNVERIFIED — citation verification failed; treat any findings as unconfirmed."

    overlapping = [c for c in assessment.element_comparisons if c.overlap_assessed]
    if not overlapping:
        return f"Candidate {assessment.candidate_patent_id}: VERIFIED — no substantial overlap found."

    findings = "; ".join(f'"{c.disclosure_element}" overlaps claim {c.candidate_claim_number} ({c.overlap_explanation})' for c in overlapping)
    return f"Candidate {assessment.candidate_patent_id}: VERIFIED — {findings}"


def _build_user_prompt(disclosure: InventionDisclosure, assessments: list[NoveltyAssessment]) -> str:
    if not assessments:
        return f"Invention disclosure technical field: {disclosure.technical_field}\n\nNo candidate prior art was assessed."

    assessment_lines = "\n".join(_describe_assessment(a) for a in assessments)
    return f"Invention disclosure technical field: {disclosure.technical_field}\n\nCandidate assessments:\n{assessment_lines}"


@traced("risk_report_agent")
def generate_risk_report(
    disclosure: InventionDisclosure,
    assessments: list[NoveltyAssessment],
    client: RotatingChatGroq | None = None,
    settings: Settings | None = None,
) -> FTOReport:
    """Aggregate already citation-verified `assessments` into an `FTOReport`.

    Callers are responsible for running `citation_guard.verify_citations()` on each
    assessment before calling this — this agent trusts `citation_verified` as given, it
    doesn't re-derive it.
    """
    settings = settings or get_settings()
    client = client or build_groq_client(settings)

    response = client.invoke(
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(disclosure, assessments)},
        ]
    )

    return FTOReport(disclosure=disclosure, assessments=assessments, summary=response.content)
