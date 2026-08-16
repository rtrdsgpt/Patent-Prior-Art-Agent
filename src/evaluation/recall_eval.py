"""Recall@k evaluation against real examiner-cited prior art (todo.md section 5 — "the
differentiator... a rigorous, defensible eval design, not just an LLM-judge score").

**Eval query construction:** there's no dataset of real free-text invention disclosures with
known correct answers, so each ingested patent stands in as its own query: its title +
abstract is used as a synthetic "disclosure," and its own real examiner-cited references
(`Patent.examiner_cited_patent_ids` — see `ingestion/bigquery_client.py`'s "EXA vs SEA" log
entry for how this ground truth was actually confirmed to exist) are the ground-truth
relevant set. This is the standard "citation prediction" / leave-one-out eval design for
prior-art search: real ground truth, no synthetic disclosure-writing or LLM-judge needed.

**Scores raw retrieval (`retrieval/hybrid.py` + `retrieval/reranker.py`), not the full agent
pipeline.** recall@k is conventionally a retrieval-quality metric — it scores a ranked
document list against relevance judgments, which is what `hybrid_search`/`rerank` produce
directly. Routing every eval query through `agents/disclosure_parser.py` and
`agents/search_agent.py` first would (a) make disclosure-parsing quality a confound in what's
supposed to be a retrieval-quality signal, and (b) cost one-to-several Groq calls per eval
case, which would force evaluating only a small, cost-constrained sample instead of the full
eval set. Using title+abstract directly as the query keeps this fast, free, deterministic,
and runnable over every case — matching what section 4's MLflow bullet actually wants to
track ("retrieval/reranking experiments: embedding models, chunking strategies"). Evaluating
the disclosure-parser/search-agent's own contribution to end-to-end quality is a real, valid,
different question — just not what recall@k against examiner citations answers.

**Two metrics, not one, because most citations are unreachable by design.** This corpus is
deliberately a single-CPC-class, ~300-patent slice (`docs/cpc_scope.md`'s "scope
discipline") — most of a patent's real examiner citations point outside it entirely (older
patents, different CPC classes, non-patent literature). Scoring recall@k against *all*
citations honestly reports a low number that mostly reflects corpus coverage, not retrieval
quality. Scoring recall@k against only the subset of citations that are actually present in
the indexed corpus (`in_corpus`) isolates retrieval-algorithm quality from that confound.
Reporting only the first would understate what the retrieval stack itself is doing; reporting
only the second would quietly hide the corpus-size limitation. Both, clearly labeled, is the
honest version.
"""

from __future__ import annotations

from dataclasses import dataclass

from chromadb.api.models.Collection import Collection
from grounded_evals import RetrievalEvalReport, evaluate_retrieval

from config.settings import Settings, get_settings
from retrieval.bm25_index import BM25Index
from retrieval.hybrid import hybrid_search
from retrieval.reranker import rerank
from schema import Patent


@dataclass
class EvalCase:
    patent_id: str
    query_text: str
    relevant_patent_ids: set[str]
    in_corpus_relevant_patent_ids: set[str]


def build_eval_set(patents: list[Patent]) -> list[EvalCase]:
    """One `EvalCase` per patent that has at least one real examiner citation — patents
    with none can't contribute a recall@k signal (there's nothing to have found)."""
    corpus_ids = {p.patent_id for p in patents}
    cases = []
    for patent in patents:
        relevant = patent.examiner_cited_patent_ids
        if not relevant:
            continue
        cases.append(
            EvalCase(
                patent_id=patent.patent_id,
                query_text=f"{patent.title}. {patent.abstract}",
                relevant_patent_ids=relevant,
                in_corpus_relevant_patent_ids=relevant & corpus_ids,
            )
        )
    return cases


@dataclass
class RecallEvalResult:
    k: int
    num_cases: int
    overall: RetrievalEvalReport
    num_in_corpus_cases: int
    in_corpus: RetrievalEvalReport | None


def run_recall_eval(
    eval_cases: list[EvalCase],
    bm25_index: BM25Index,
    embedding_collection: Collection,
    patents_by_id: dict[str, Patent],
    settings: Settings | None = None,
    k: int = 10,
    sample_size: int | None = None,
) -> RecallEvalResult:
    """Run hybrid search + rerank for every eval case and score recall@k/MRR/nDCG@k.

    `sample_size` caps how many cases run, for a quick dev-loop check — the full set has no
    LLM cost gating it (see module docstring), so there's no forced reason to sample by
    default; `None` runs everything.
    """
    settings = settings or get_settings()
    cases = eval_cases[:sample_size] if sample_size else eval_cases

    # Each query is one corpus patent's own title+abstract (see module docstring), so that
    # patent trivially self-matches and would otherwise occupy a top-k slot with a
    # non-informative hit. Fetch one extra candidate so filtering the self-match out still
    # leaves a true top-k of *other* patents, then truncate to k explicitly — recall@k, MRR,
    # and nDCG@k all need to see the same correctly-sized list to stay consistent with each
    # other (grounded_evals.reciprocal_rank in particular has no k bound of its own).
    eval_settings = settings.model_copy(update={"rerank_top_k": k + 1})

    retrieved_by_case: dict[str, list[str]] = {}
    for case in cases:
        candidates = hybrid_search(bm25_index, embedding_collection, case.query_text, settings=eval_settings)
        reranked = rerank(case.query_text, candidates, patents_by_id, settings=eval_settings)
        retrieved_by_case[case.patent_id] = [r.patent_id for r in reranked if r.patent_id != case.patent_id][:k]

    overall = evaluate_retrieval(
        [retrieved_by_case[c.patent_id] for c in cases],
        [c.relevant_patent_ids for c in cases],
        k=k,
    )

    in_corpus_cases = [c for c in cases if c.in_corpus_relevant_patent_ids]
    in_corpus = None
    if in_corpus_cases:
        in_corpus = evaluate_retrieval(
            [retrieved_by_case[c.patent_id] for c in in_corpus_cases],
            [c.in_corpus_relevant_patent_ids for c in in_corpus_cases],
            k=k,
        )

    return RecallEvalResult(k=k, num_cases=len(cases), overall=overall, num_in_corpus_cases=len(in_corpus_cases), in_corpus=in_corpus)
