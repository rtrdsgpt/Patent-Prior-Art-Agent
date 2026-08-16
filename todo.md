# Patent Prior-Art / Freedom-to-Operate Agent — TODO

New greenfield project: given an invention disclosure, find and analyze real prior art from actual
patent data and produce a grounded novelty/infringement-risk report. This folder is currently
empty — this is your Agentic/GenAI/LLM CV flagship. See `Project Plan.md` (Projects root)
section 4 for full architecture rationale.

## 0. Data source setup
- [x] ~~Register/test access to the **USPTO PatentsView API**~~ — superseded: switched to
      **Google Patents Public Data (BigQuery)** instead, which already has claims text,
      abstracts, CPC codes, assignees, and (crucially) examiner-cited references without a
      separate live API integration. See `docs/cpc_scope.md` and `README.md` for the
      reasoning, and `log.md`'s 2026-08-16 entries for how this was actually verified live
      (including a correction: the field's documented `EXA` citation category turned out to
      never be populated in practice — `SEA` is the real examiner-citation signal).
- [x] Pick one narrow CPC class to scope the first working slice — `G06N3`, see
      `docs/cpc_scope.md`.

## 1. Ingestion / RAG index (build this before the agents)
- [x] Download/ingest a corpus subset of patents in the target CPC class — real BigQuery
      ingestion (`ingestion/bigquery_client.py`, `ingestion/ingest_corpus.py`), 291 patents
      cached to `data/corpus.json` (gitignored; see section 4's DVC item).
- [x] Claim-level chunking (a natural, legally meaningful chunk boundary) —
      `ingestion/chunking.py`.
- [x] Embed + index in a vector DB (Chroma) — `retrieval/embedding_index.py`.
- [x] Hybrid retrieval: BM25 + embeddings — `retrieval/bm25_index.py` +
      `retrieval/hybrid.py` (Reciprocal Rank Fusion).
- [x] Cross-encoder reranking stage before the comparison agent —
      `retrieval/reranker.py`.

## 2. Agents
- [x] **Disclosure-parser agent** — `agents/disclosure_parser.py`.
- [x] **Prior-art search agent** — `agents/search_agent.py`; adaptive query expansion is a
      bounded 3-query sequence (medium/broad/narrow) built from the disclosure-parser's
      output, not another LLM call — see log.md for why.
- [x] **Claims-parser agent** — `agents/claims_parser.py`.
- [x] **Comparison/novelty-assessment agent** — `agents/comparison_agent.py`.
- [x] **Risk-report/critic agent** — `agents/risk_report_agent.py`; citation-verification
      guard is its own deterministic module, `agents/citation_guard.py` (no LLM).
- [x] Orchestrate agents — `agents/orchestrator.py`, a hand-rolled bounded sequence (not
      LangGraph — see that module's docstring for why).

## 3. API layer
- [x] FastAPI: `POST /disclosure/analyze` → job id
- [x] `GET /jobs/{id}` for status
- [x] `GET /report/{id}` for the FTO report — returns a real `FTOReport`.

## 4. MLOps
- [x] Docker/docker-compose: app + vector DB — `Dockerfile`/`docker-compose.yml`, verified
      end-to-end. (Skipped the "optional local embedding server" — embedding already runs
      in-process, so a dedicated service has no consumer.)
- [ ] **Airflow**: scheduled incremental ingestion DAG for new patents in the target CPC classes
      (download → clean → chunk → embed → index)
- [ ] **MLflow**: track retrieval/reranking experiments (embedding models, chunking strategies)
      and log report-generation runs
- [ ] **DVC**: version the ingested patent corpus subset and embedding indices

## 5. Evaluation (the differentiator — real ground truth)
- [ ] Build an eval set from patents with known examiner-cited prior art — now unblocked:
      149/291 ingested patents carry real examiner (`SEA`-category) citations.
- [ ] Run the search+comparison pipeline on those disclosures, score **recall@k** against the
      actual examiner citations — a rigorous, defensible eval design, not just an LLM-judge score

## 6. Tracing
- [ ] OpenTelemetry/Langfuse spans per agent step (search → parse → compare → verify)
- [ ] Explicitly trace the citation-verification guard as a checked step

## 7. Testing
- [x] pytest for the claims-parser's structured output — `tests/test_claims_parser.py`.
- [x] pytest for the citation-verification guard — `tests/test_citation_guard.py`.
- [x] pytest for chunking logic — `tests/test_chunking.py`.
- [x] Integration test over a small fixed patent set with a mocked LLM —
      `tests/test_orchestrator.py`'s non-live tests run the full orchestrator over the
      fixture corpus with every agent function mocked; per-agent unit tests
      (`test_disclosure_parser.py` etc.) additionally mock at the Groq-client level. 145
      tests total (`pytest -m slow`/`-m integration` opt-in for real models/live APIs).

## 8. MCP
- [x] Expose `search-prior-art` as an MCP tool — `mcp_server.py`.
- [x] Expose `assess-novelty` as an MCP tool — `mcp_server.py`.
