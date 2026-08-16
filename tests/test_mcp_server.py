import pytest

from mcp_server import mcp
from schema import SearchResult

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


async def test_search_prior_art_is_registered():
    tools = await mcp.list_tools()
    assert {t.name for t in tools} == {"search_prior_art", "assess_novelty"}


async def test_search_prior_art_returns_candidates(monkeypatch):
    monkeypatch.setattr(
        "mcp_server.run_prior_art_search",
        lambda disclosure_text: [SearchResult(patent_id="US10000001B2", score=0.9, retrieval_method="reranked")],
    )
    result = await mcp.call_tool("search_prior_art", {"disclosure_text": "a neural network", "top_k": 5})
    assert result.is_error is False
    assert result.structured_content["result"][0]["patent_id"] == "US10000001B2"


async def test_search_prior_art_respects_top_k(monkeypatch):
    candidates = [SearchResult(patent_id=f"P{i}", score=1.0 - i * 0.1, retrieval_method="reranked") for i in range(5)]
    monkeypatch.setattr("mcp_server.run_prior_art_search", lambda disclosure_text: candidates)
    result = await mcp.call_tool("search_prior_art", {"disclosure_text": "x", "top_k": 2})
    assert len(result.structured_content["result"]) == 2


async def test_assess_novelty_raises_not_implemented_tool_error():
    from mcp.server.mcpserver.exceptions import ToolError

    with pytest.raises(ToolError, match="not yet implemented"):
        await mcp.call_tool("assess_novelty", {"disclosure_text": "x", "candidate_patent_id": "US10000001B2"})
