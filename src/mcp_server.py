"""MCP server exposing the pipeline as tools — todo.md section 8.

Both tools are real now that the full agent pipeline exists: `search_prior_art` runs hybrid
search + rerank, `assess_novelty` runs the claims-parser → comparison → citation-guard chain
against one named candidate patent.
"""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from api.pipeline import run_novelty_assessment, run_prior_art_search

mcp = MCPServer("patent-prior-art-agent")


@mcp.tool()
def search_prior_art(disclosure_text: str, top_k: int = 10) -> list[dict]:
    """Search the indexed patent corpus for prior-art candidates relevant to a free-text
    invention disclosure. Returns up to `top_k` candidates, hybrid-retrieved and
    cross-encoder reranked, ordered most-relevant first."""
    candidates = run_prior_art_search(disclosure_text)
    return [c.model_dump(mode="json") for c in candidates[:top_k]]


@mcp.tool()
def assess_novelty(disclosure_text: str, candidate_patent_id: str) -> dict:
    """Assess element-by-element novelty overlap between a disclosure and one specific
    candidate patent's independent claims (get `candidate_patent_id` from
    `search_prior_art`). The result's `citation_verified` field indicates whether every
    quoted claim text was independently confirmed to genuinely appear in the patent — check
    it before treating `element_comparisons` as grounded."""
    try:
        assessment = run_novelty_assessment(disclosure_text, candidate_patent_id)
    except KeyError:
        raise ToolError(f"No candidate patent {candidate_patent_id!r} in the indexed corpus. Use search_prior_art to find valid candidate_patent_id values.") from None

    return assessment.model_dump(mode="json")


if __name__ == "__main__":
    mcp.run()
