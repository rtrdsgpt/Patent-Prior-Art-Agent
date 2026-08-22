"""Orchestrates the full section-2 pipeline as a LangGraph `StateGraph`: disclosure-parser
→ search → [claims-parser + comparison + citation-guard, fanned out per candidate] →
risk-report.

**LangGraph, not a hand-rolled sequence** (an earlier version of this module was
deliberately hand-rolled — see log.md for that reasoning and why it changed). The
per-candidate step is a genuine map-reduce: every candidate is assessed independently and the
results are reduced back into one `assessments` list before risk-reporting runs — LangGraph's
`Send` API models that directly (dynamic fan-out over a runtime-sized list, with the
downstream node only firing once every fan-out branch for that step has completed), instead
of a plain Python `for` loop pretending there's no real branching happening. Verified this
mechanism with a standalone toy graph (`Send` + an `Annotated[list, operator.add]` reducer)
before wiring in the real agents, not assumed from the docs.

**One `RotatingChatGroq` (LangChain's Groq chat model, wrapped for multi-key rotation — see
`agents/groq_client.py`) built once and closed over by every node**, not stored in graph
state — `agents/groq_client.py`'s key rotation keeps state (`_current`) on the client
instance, so if every node built its own client, each would restart rotation from key 0
independently, defeating the point of spreading load across keys over one pipeline run. The
retrieval indexes (`bm25_index`/`embedding_collection`/`patents_by_id`) are bound the same
way, for the more ordinary reason that they're large, non-serializable objects that don't
belong in graph state that LangGraph may checkpoint.
"""

from __future__ import annotations

import logging
import operator
from typing import Annotated, TypedDict

from chromadb.api.models.Collection import Collection
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from opentelemetry import trace as otel_trace

from agents.citation_guard import verify_citations
from agents.claims_parser import parse_claim_elements
from agents.comparison_agent import assess_novelty
from agents.disclosure_parser import parse_disclosure
from agents.groq_client import RotatingChatGroq, build_groq_client
from agents.risk_report_agent import generate_risk_report
from agents.search_agent import search_prior_art
from config.settings import Settings, get_settings
from experiment_tracking import log_report_run
from retrieval.bm25_index import BM25Index
from schema import FTOReport, InventionDisclosure, NoveltyAssessment, Patent, SearchResult
from tracing import traced

logger = logging.getLogger(__name__)


class PipelineState(TypedDict, total=False):
    disclosure_text: str
    disclosure: InventionDisclosure
    candidates: list[SearchResult]
    # operator.add reducer: each per-candidate branch contributes its own one-element list,
    # LangGraph merges them back into a single list once every fanned-out branch completes.
    assessments: Annotated[list[NoveltyAssessment], operator.add]
    report: FTOReport


class _CandidateState(TypedDict):
    """The per-branch state `Send` hands to `assess_candidate` — deliberately narrower than
    `PipelineState`: this node only ever needs one candidate plus the shared disclosure."""

    disclosure: InventionDisclosure
    candidate: SearchResult


def _build_graph(
    bm25_index: BM25Index,
    embedding_collection: Collection,
    patents_by_id: dict[str, Patent],
    client: RotatingChatGroq,
    settings: Settings,
):
    def disclosure_parser_node(state: PipelineState) -> dict:
        disclosure = parse_disclosure(state["disclosure_text"], client=client, settings=settings)
        return {"disclosure": disclosure}

    def search_node(state: PipelineState) -> dict:
        candidates = search_prior_art(state["disclosure"], bm25_index, embedding_collection, patents_by_id, settings=settings)
        otel_trace.get_current_span().set_attribute("num_candidates", len(candidates))
        return {"candidates": candidates}

    def route_to_candidates(state: PipelineState) -> list[Send] | str:
        if not state["candidates"]:
            return "risk_report"
        return [Send("assess_candidate", {"disclosure": state["disclosure"], "candidate": c}) for c in state["candidates"]]

    def assess_candidate_node(state: _CandidateState) -> dict:
        candidate = state["candidate"]
        patent = patents_by_id.get(candidate.patent_id)
        if patent is None:
            # Shouldn't happen — candidates come from indexes built off patents_by_id — but
            # a stale/mismatched index shouldn't take down the whole pipeline over one entry.
            logger.warning("Search returned patent_id %r not found in patents_by_id; skipping", candidate.patent_id)
            return {"assessments": []}

        claim_elements = parse_claim_elements(patent, client=client, settings=settings)
        assessment = assess_novelty(state["disclosure"], patent, claim_elements=claim_elements, client=client, settings=settings)
        return {"assessments": [verify_citations(assessment, patent)]}

    def risk_report_node(state: PipelineState) -> dict:
        assessments = state.get("assessments", [])
        otel_trace.get_current_span().set_attribute("num_assessments", len(assessments))
        report = generate_risk_report(state["disclosure"], assessments, client=client, settings=settings)
        log_report_run(report, settings)
        return {"report": report}

    graph = StateGraph(PipelineState)
    graph.add_node("disclosure_parser", disclosure_parser_node)
    graph.add_node("search", search_node)
    graph.add_node("assess_candidate", assess_candidate_node)
    graph.add_node("risk_report", risk_report_node)

    graph.add_edge(START, "disclosure_parser")
    graph.add_edge("disclosure_parser", "search")
    graph.add_conditional_edges("search", route_to_candidates, ["assess_candidate", "risk_report"])
    graph.add_edge("assess_candidate", "risk_report")
    graph.add_edge("risk_report", END)

    return graph.compile()


@traced("fto_pipeline")
def run_fto_pipeline(
    disclosure_text: str,
    bm25_index: BM25Index,
    embedding_collection: Collection,
    patents_by_id: dict[str, Patent],
    client: RotatingChatGroq | None = None,
    settings: Settings | None = None,
) -> FTOReport:
    """Run the full disclosure → FTO report pipeline end to end.

    `@traced` makes this the root span for a run — every agent stage below (each already
    individually `@traced`) nests under it automatically via `start_as_current_span`'s
    context propagation, giving a trace viewer one tree per pipeline run.
    """
    settings = settings or get_settings()
    client = client or build_groq_client(settings)

    graph = _build_graph(bm25_index, embedding_collection, patents_by_id, client, settings)
    final_state = graph.invoke({"disclosure_text": disclosure_text, "assessments": []})
    return final_state["report"]
