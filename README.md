# Patent Prior-Art Agent

Given a free-text invention disclosure, find and analyze real prior art from actual patent
data and produce a grounded novelty/freedom-to-operate report — with every claim in the
report citing specific, verified source text, and evaluated against real ground truth
(examiner-cited prior art), not just an LLM-judge score.

## Status: early scaffolding

This is a greenfield build. What exists so far:

- **Data models** (`src/patent_agent/schema.py`) — `Patent`, `Claim`, `InventionDisclosure`,
  `SearchResult`, `ClaimElementComparison`, `NoveltyAssessment`, `FTOReport`.
- **Claim-level chunking** (`src/patent_agent/ingestion/chunking.py`) — parses a patent's
  raw claims-section text into individual `Claim` objects (claim number, text,
  independent/dependent, what it depends on). Claim-level is the chunk boundary used
  throughout, since a claim is the smallest independently legally-meaningful unit.
- **Settings** (`src/patent_agent/config/settings.py`) — env-driven config (Groq for the
  LLM, BigQuery for ingestion, retrieval/reranking knobs).
- **CPC scope decision** (`docs/cpc_scope.md`) — first working slice is scoped to `G06N3`
  (neural networks); see that doc for why.

Not yet built: BigQuery ingestion against Google Patents Public Data, hybrid
BM25+dense retrieval, the reranker, the agent pipeline (disclosure-parser → search →
claims-parser → comparison → risk-report), the FastAPI layer, and the examiner-citation
recall@k evaluation harness. Full plan in [`todo.md`](todo.md).

## Data source

[Google Patents Public Data](https://console.cloud.google.com/marketplace/product/google_patents_public_datasets/google-patents-public-data)
(BigQuery) — chosen over the live USPTO API because it already contains USPTO's own
examiner-cited references (`citation.category = "EXA"`), structured and bulk-queryable via
SQL, without the account-creation friction of the (now-authenticated-only) USPTO Open Data
Portal. Same underlying data, no separate live API integration needed.

## Evaluation

Ground truth for prior-art recall is real: patents in the target CPC class already have
known examiner-cited prior art, so the search+comparison pipeline can be scored with
recall@k against actual examiner citations rather than relying on a subjective LLM-judge
score alone. Retrieval/citation-verification metrics are shared with the rest of the
portfolio via [`grounded-evals`](https://github.com/rtrdsgpt/grounded-evals).

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env  # fill in GROQ_API_KEY, GCP_PROJECT_ID
pytest
```
