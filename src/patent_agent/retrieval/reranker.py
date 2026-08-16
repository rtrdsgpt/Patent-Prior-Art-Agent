"""Cross-encoder reranking stage, run on the hybrid retriever's candidate list before the
comparison agent sees it.

BM25/dense retrieval score a query against each chunk independently (bi-encoder style),
which scales to a large corpus but can't model query-document interaction directly. A
cross-encoder scores the query and a candidate's text jointly in one forward pass — much
more accurate, but too slow to run over an entire corpus, hence it only runs over the
hybrid stage's already-narrowed candidate list. See todo.md section 1.
"""

from __future__ import annotations

from sentence_transformers import CrossEncoder

from patent_agent.config.settings import Settings, get_settings
from patent_agent.schema import Patent, SearchResult


def _patent_document_text(patent: Patent) -> str:
    """The text a candidate patent is reranked against — abstract plus independent claims,
    since those two together are the most representative single blob of what the patent
    actually claims, without pulling in every dependent claim's narrower detail."""
    independent_claims_text = " ".join(c.text for c in patent.independent_claims)
    return f"{patent.abstract} {independent_claims_text}".strip()


def rerank(
    query: str,
    candidates: list[SearchResult],
    patents_by_id: dict[str, Patent],
    settings: Settings | None = None,
    model: CrossEncoder | None = None,
) -> list[SearchResult]:
    """Rerank `candidates` with a cross-encoder, returning the top `settings.rerank_top_k`.

    `model` is accepted as a parameter (rather than always constructed internally) so
    callers running many reranks in a session can load the cross-encoder once and reuse it —
    it's a real model load, not a cheap lookup.
    """
    settings = settings or get_settings()
    model = model or CrossEncoder(settings.reranker_model)

    scoreable = [c for c in candidates if c.patent_id in patents_by_id]
    if not scoreable:
        return []

    pairs = [(query, _patent_document_text(patents_by_id[c.patent_id])) for c in scoreable]
    scores = model.predict(pairs)

    reranked = sorted(zip(scoreable, scores), key=lambda pair: pair[1], reverse=True)
    return [
        SearchResult(patent_id=candidate.patent_id, score=float(score), retrieval_method="reranked")
        for candidate, score in reranked[: settings.rerank_top_k]
    ]
