# Patent Prior-Art Agent

Given a free-text invention disclosure, find and analyze real prior art from actual patent
data and produce a grounded novelty/freedom-to-operate report — with every claim in the
report citing specific, verified source text, and evaluated against real ground truth
(examiner-cited prior art), not just an LLM-judge score.

## Status: full pipeline built and working end to end

Every stage of [`todo.md`](todo.md) is implemented except the MLOps items still marked open
there (Airflow's own live scheduler bring-up, MLflow's UI). Full build history, every design
decision, every real bug found and how it was fixed, is in [`log.md`](log.md) — written for
interview prep, not just a changelog.

- **Ingestion** (`src/ingestion/`) — real BigQuery ingestion from Google Patents Public Data,
  seeded by CPC class then expanded along citation edges (`fetch_patents_by_id`) so examiner
  citations are actually reachable within the indexed corpus, not just referenced by ID.
- **Retrieval** (`src/retrieval/`) — claim-level chunking, BM25 + dense embedding hybrid
  search (Reciprocal Rank Fusion), cross-encoder reranking.
- **Agents** (`src/agents/`) — disclosure-parser, prior-art search (adaptive query
  expansion), claims-parser, comparison/novelty-assessment, risk-report, and a deterministic
  citation-verification guard, orchestrated with **LangGraph** (`agents/orchestrator.py`'s
  `StateGraph`, with a real `Send`-based dynamic fan-out for the per-candidate step). All LLM
  calls go through **LangChain**'s `ChatGroq`, wrapped for multi-key rotation
  (`agents/groq_client.py`).
- **API** (`src/api/`) — FastAPI: `POST /disclosure/analyze` → job id, `GET /jobs/{id}`,
  `GET /report/{id}` → a real `FTOReport`.
- **MCP** (`src/mcp_server.py`) — `search_prior_art` and `assess_novelty` tools.
- **Tracing** (`src/tracing.py`) — OpenTelemetry spans per pipeline stage, console exporter.
- **Experiment tracking** (`src/experiment_tracking.py`) — MLflow runs for both retrieval
  eval runs and report-generation runs (local SQLite-backed store).
- **Evaluation** (`src/evaluation/`) — recall@k/MRR/nDCG@k against real examiner citations,
  not an LLM-judge score (see below).
- **Docker/docker-compose** — app + Chroma + a scheduled ingestion DAG (Airflow).
- **DVC** — `data/corpus.json` and `chroma_db/` are DVC-tracked.

## Data source

[Google Patents Public Data](https://console.cloud.google.com/marketplace/product/google_patents_public_datasets/google-patents-public-data)
(BigQuery) — chosen over the live USPTO API because it's structured and bulk-queryable via
SQL, without the account-creation friction of the (now-authenticated-only) USPTO Open Data
Portal. One correction worth flagging: the field's own documentation lists `EXA` as the
examiner-citation category, but that value is never actually populated in the live table —
`SEA` (DOCDB's "search report" category) is the real signal. Verified live before writing
the ingestion query; see `log.md`'s 2026-08-16 entries for the exact `bq query` that found
this.

## Evaluation

Ground truth is real: patents already carry known examiner-cited prior art, so retrieval is
scored with recall@k/MRR/nDCG@k against actual citations, not an LLM-judge score. Two
numbers are reported, not one — most citations point outside this deliberately small,
single-CPC-class corpus, so an "overall" score alone would mostly measure corpus coverage,
not retrieval quality:

```
=== Recall@10 eval — 811 cases ===
Overall (against ALL real examiner citations, including ones outside this 1488-patent corpus):
  recall@10: 0.087   MRR: 0.144   nDCG@10: 0.093

In-corpus only (267 cases with a citation actually present in the index):
  recall@10: 0.372   MRR: 0.438   nDCG@10: 0.330
```

The low "overall" number is expected and honest (corpus coverage, not a bug); the
"in-corpus" number is the real signal for whether hybrid search + reranking works — see
`src/evaluation/recall_eval.py`'s docstring for the full design reasoning. Run it yourself:

```bash
python -m evaluation.run_eval --k 10 [--sample-size N]
```

Retrieval/citation-verification scoring primitives are shared with the rest of the CV
portfolio via [`grounded-evals`](https://github.com/rtrdsgpt/grounded-evals).

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env  # fill in GROQ_API_KEY, GCP_PROJECT_ID
pytest                # fast unit tests; add -m slow for real local-model tests,
                       # -m integration for tests that hit live BigQuery/Groq
```

**Ingest a real corpus** (needs `GCP_PROJECT_ID` + `gcloud auth application-default login`):

```bash
python -m ingestion.ingest_corpus   # writes data/corpus.json (DVC-tracked, see below)
```

**Run the API locally:**

```bash
uvicorn api.app:app --reload
```

**Or the full stack** (app + Chroma + the Airflow ingestion DAG):

```bash
docker compose up --build
# app:      http://localhost:8080
# chroma:   http://localhost:8100
# airflow:  http://localhost:8081
```

**Data versioning** — `data/corpus.json` and `chroma_db/` are tracked with DVC (see
`.dvc/config`, `data/corpus.json.dvc`):

```bash
dvc pull   # fetch the versioned corpus/index instead of re-ingesting from scratch
dvc push   # after regenerating either
```

**Experiment tracking** — every eval run and report-generation run is logged to a local
MLflow store (`sqlite:///mlflow.db`):

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
