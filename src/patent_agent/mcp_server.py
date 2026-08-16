"""MCP server exposing the pipeline as tools — todo.md section 8.

Same honesty principle as the FastAPI `/report` route (see `api/app.py`'s docstring):
`search_prior_art` is real (it's the same hybrid-search-plus-rerank pipeline the API uses),
`assess_novelty` raises a clear `ToolError` rather than faking a comparison, since the
comparison agent it would depend on (todo.md section 2) is paused pending a Groq API key.
"""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from patent_agent.api.pipeline import run_prior_art_search

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
    """Assess element-by-element novelty overlap between a disclosure and a candidate
    patent's claims. Not yet available: this depends on the comparison agent (todo.md
    section 2), which is paused pending a Groq API key — see the project's log.md."""
    raise ToolError(
        "assess_novelty is not yet implemented: it depends on the comparison agent "
        "(todo.md section 2), which is paused pending a Groq API key. Use "
        "search_prior_art for candidate retrieval in the meantime."
    )


if __name__ == "__main__":
    mcp.run()
