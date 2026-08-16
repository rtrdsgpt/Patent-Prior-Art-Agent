# Patent Prior-Art / Freedom-to-Operate Agent — TODO

Given an invention disclosure, find and analyze real prior art from actual patent data and
produce a grounded novelty/infringement-risk report. See `Project Plan.md` (Projects root)
section 4 for full architecture rationale. **All sections below are now built** — see
`README.md` for the current system overview and `log.md` for the full build history/
reasoning behind every decision.

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
      ingestion (`ingestion/bigquery_client.py`, `ingestion/ingest_corpus.py`). Grew from an
      initial 291-patent CPC-scoped seed to 1488 patents once citation-aware expansion
      (`fetch_patents_by_id`) landed — see section 5, the seed alone had ~0% chance of
      containing any patent's own cited prior art. Cached to `data/corpus.json`, DVC-tracked
      (section 4).
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
- [x] **Airflow**: scheduled ingestion DAG — `dags/ingest_corpus_dag.py`
      (`ingest_corpus >> embed_and_index`), own container (`Dockerfile.airflow`), verified
      with a real `airflow dags test` run against live BigQuery, not just a successful image
      build. "Incremental" means re-running the same bounded fetch on a schedule, not a true
      watermark-based delta load — see the DAG's own docstring for why.
- [x] **MLflow**: `experiment_tracking.py` — every eval run (`run_eval.py`) and every real
      report-generation run (`orchestrator.py`) logged to a local SQLite-backed store.
- [x] **DVC**: `data/corpus.json` tracked (`.dvc/config`, local remote,
      `data/corpus.json.dvc`); `chroma_db/` set up the same way, tracked once a persistent
      index exists on disk.

## 5. Evaluation (the differentiator — real ground truth)
- [x] Build an eval set from patents with known examiner-cited prior art —
      `evaluation/recall_eval.py`'s `build_eval_set()`.
- [x] Run search, score **recall@k** against actual examiner citations —
      `evaluation/run_eval.py` (`python -m evaluation.run_eval`). Real results over the full
      1488-patent corpus, 811 eval cases: overall recall@10 0.087 (honest — reflects corpus
      coverage, most citations point outside this deliberately small corpus), in-corpus
      recall@10 0.372 / MRR 0.438 (the real retrieval-quality signal, isolated from corpus
      coverage). Full reasoning for the two-metric design in that module's docstring.

## 6. Tracing
- [x] OpenTelemetry spans per agent step — `tracing.py`'s `@traced`, applied to every stage;
      console exporter (no Langfuse/OTLP backend credentials available, but real/inspectable
      on its own — see `tracing.py`'s docstring for swapping in a real backend later).
- [x] Citation-verification guard explicitly traced as a checked step —
      `agents/citation_guard.py` sets `patent_id`/`num_comparisons_checked`/
      `citation_verified` as span attributes directly, not just a generic pass/fail span.

## 7. Testing
- [x] pytest for the claims-parser's structured output — `tests/test_claims_parser.py`.
- [x] pytest for the citation-verification guard — `tests/test_citation_guard.py`.
- [x] pytest for chunking logic — `tests/test_chunking.py`.
- [x] Integration test over a small fixed patent set with a mocked LLM —
      `tests/test_orchestrator.py`'s non-live tests run the full orchestrator over the
      fixture corpus with every agent function mocked; per-agent unit tests
      (`test_disclosure_parser.py` etc.) additionally mock at the Groq-client level. 172
      tests total (`pytest -m slow`/`-m integration` opt-in for real models/live APIs; 1
      Airflow-DAG test skips outside that image by design — see `log.md`).

## 8. MCP
- [x] Expose `search-prior-art` as an MCP tool — `mcp_server.py`.
- [x] Expose `assess-novelty` as an MCP tool — `mcp_server.py`.
