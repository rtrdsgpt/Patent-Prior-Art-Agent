"""Core data models shared across ingestion, retrieval, and the agent pipeline.

Pydantic models throughout so agent outputs (structured LLM responses) validate the same
way ingested data does.
"""

from __future__ import annotations

from datetime import date
from enum import Enum

from pydantic import BaseModel, Field


class CitationCategory(str, Enum):
    """Subset of Google Patents Public Data's `citation.category` values relevant here.

    EXAMINER is the one that matters for ground-truth eval (section 5 of todo.md) — it's
    prior art the examiner themselves found and cited, not just something the applicant
    listed. See docs/cpc_scope.md for why this field is what makes the eval possible.
    """

    EXAMINER = "EXA"
    APPLICANT = "APP"
    OTHER = "OTHER"


class Citation(BaseModel):
    cited_patent_id: str
    category: CitationCategory


class Claim(BaseModel):
    """One claim of a patent. Claim-level is the chunk boundary used throughout —
    see ingestion/chunking.py for why."""

    claim_number: int
    text: str
    is_independent: bool
    depends_on: int | None = Field(default=None, description="Claim number this claim depends on, if not independent")


class Patent(BaseModel):
    patent_id: str
    title: str
    abstract: str
    claims: list[Claim]
    cpc_codes: list[str]
    assignees: list[str]
    publication_date: date | None = None
    citations: list[Citation] = Field(default_factory=list)

    @property
    def examiner_cited_patent_ids(self) -> set[str]:
        """Ground-truth prior-art set for this patent — used to build the eval set."""
        return {c.cited_patent_id for c in self.citations if c.category == CitationCategory.EXAMINER}

    @property
    def independent_claims(self) -> list[Claim]:
        return [c for c in self.claims if c.is_independent]


class InventionDisclosure(BaseModel):
    """Free-text input to the pipeline, plus the disclosure-parser agent's structured
    extraction of it (technical_field/key_elements/candidate_cpc_classes start unset and
    are filled in by that agent — see agents/disclosure_parser.py)."""

    raw_text: str
    technical_field: str | None = None
    key_elements: list[str] = Field(default_factory=list)
    candidate_cpc_classes: list[str] = Field(default_factory=list)


class SearchResult(BaseModel):
    patent_id: str
    score: float
    retrieval_method: str = Field(description='e.g. "bm25", "dense", "hybrid", "reranked"')


class ClaimElementComparison(BaseModel):
    """One element-by-element overlap assessment between a disclosure element and a
    candidate patent's claim — the atomic unit of the comparison agent's output, and what
    the citation-verification guard checks against source claim text."""

    disclosure_element: str
    candidate_patent_id: str
    candidate_claim_number: int
    cited_claim_text: str
    overlap_explanation: str
    overlap_assessed: bool


class NoveltyAssessment(BaseModel):
    candidate_patent_id: str
    element_comparisons: list[ClaimElementComparison]
    citation_verified: bool | None = Field(default=None, description="Set by the citation-verification guard, not the LLM")


class FTOReport(BaseModel):
    disclosure: InventionDisclosure
    assessments: list[NoveltyAssessment]
    summary: str
