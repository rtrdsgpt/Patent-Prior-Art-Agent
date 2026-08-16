"""Orchestrates the full section-2 pipeline: disclosure-parser → search → [claims-parser +
comparison, per candidate] → citation guard → risk-report.

**Hand-rolled bounded sequence, not LangGraph** — todo.md offers either. There's no cyclic
or conditional control flow here that would benefit from a graph framework: every stage runs
exactly once except the per-candidate loop (a plain bounded `for`, length capped by
`settings.rerank_top_k`) and the search agent's own already-bounded internal retry
(`search_agent.py`). A linear function with early-exit guards is easier to read, test, and
step through than a graph definition for a pipeline that doesn't actually branch or loop
in an open-ended way. Revisit if orchestration grows real branching (e.g. agents disagreeing
and needing a routed resolution step) — that's the point where a graph framework starts
paying for itself.

**One `RotatingGroqClient` constructed once and threaded through every stage** — this
matters beyond avoiding repeated client construction: `agents/groq_client.py`'s key rotation
keeps state (`_current`) on the client instance. If each agent built its own client via
`build_groq_client(settings)` when none is passed, every agent would start its own rotation
from key 0 independently, defeating the point of spreading load across keys over the course
of one multi-agent pipeline run.
"""

from __future__ import annotations

import logging

from chromadb.api.models.Collection import Collection

from agents.citation_guard import verify_citations
from agents.claims_parser import parse_claim_elements
from agents.comparison_agent import assess_novelty
from agents.disclosure_parser import parse_disclosure
from agents.groq_client import RotatingGroqClient, build_groq_client
from agents.risk_report_agent import generate_risk_report
from agents.search_agent import search_prior_art
from config.settings import Settings, get_settings
from retrieval.bm25_index import BM25Index
from schema import FTOReport, Patent

logger = logging.getLogger(__name__)


def run_fto_pipeline(
    disclosure_text: str,
    bm25_index: BM25Index,
    embedding_collection: Collection,
    patents_by_id: dict[str, Patent],
    client: RotatingGroqClient | None = None,
    settings: Settings | None = None,
) -> FTOReport:
    """Run the full disclosure → FTO report pipeline end to end."""
    settings = settings or get_settings()
    client = client or build_groq_client(settings)

    disclosure = parse_disclosure(disclosure_text, client=client, settings=settings)
    candidates = search_prior_art(disclosure, bm25_index, embedding_collection, patents_by_id, settings=settings)

    assessments = []
    for candidate in candidates:
        patent = patents_by_id.get(candidate.patent_id)
        if patent is None:
            # Shouldn't happen — candidates come from indexes built off patents_by_id — but
            # a stale/mismatched index shouldn't take down the whole pipeline over one entry.
            logger.warning("Search returned patent_id %r not found in patents_by_id; skipping", candidate.patent_id)
            continue

        claim_elements = parse_claim_elements(patent, client=client, settings=settings)
        assessment = assess_novelty(disclosure, patent, claim_elements=claim_elements, client=client, settings=settings)
        assessments.append(verify_citations(assessment, patent))

    return generate_risk_report(disclosure, assessments, client=client, settings=settings)
