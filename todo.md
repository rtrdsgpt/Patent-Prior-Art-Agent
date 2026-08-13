# Patent Prior-Art / Freedom-to-Operate Agent — TODO

New greenfield project: given an invention disclosure, find and analyze real prior art from actual
patent data and produce a grounded novelty/infringement-risk report. This folder is currently
empty — this is your Agentic/GenAI/LLM CV flagship. See `Project Plan.md` (Projects root)
section 4 for full architecture rationale.

## 0. Data source setup
- [ ] Register/test access to the **USPTO PatentsView API** (free, no auth required for standard
      queries) — confirm claims text, abstracts, CPC classification codes, assignees, and
      examiner-cited prior art are all retrievable
- [ ] Pick one narrow CPC class to scope the first working slice (don't try to cover all
      classifications at once)

## 1. Ingestion / RAG index (build this before the agents)
- [ ] Download/ingest a corpus subset of patents in the target CPC class
- [ ] Claim-level chunking (a natural, legally meaningful chunk boundary)
- [ ] Embed + index in a vector DB (Qdrant or Chroma)
- [ ] Hybrid retrieval: BM25 + embeddings
- [ ] Cross-encoder reranking stage before the comparison agent

## 2. Agents
- [ ] **Disclosure-parser agent** — free-text invention disclosure → structured elements
      (technical field, key components/claims, candidate CPC classification)
- [ ] **Prior-art search agent** — CPC-class + hybrid retrieval over the indexed corpus; adaptive
      query expansion if too few/too many candidates found (reuse the bounded-retry pattern from
      Exporter Crawl's `discovery.py`)
- [ ] **Claims-parser agent** — structure each candidate patent's independent claims into discrete
      elements (conceptually similar to Legal SLM SFT's IRAC structuring, applied to patent claim
      language)
- [ ] **Comparison/novelty-assessment agent** — element-by-element, RAG-grounded overlap
      assessment vs. the disclosure, every claim citing specific source text
- [ ] **Risk-report/critic agent** — aggregate into a structured FTO-risk report; deterministic
      citation-verification guard re-checks every quoted claim string genuinely appears in the
      source patent (reuse the hallucination-guard pattern from Exporter Crawl's
      `rank_engine.py`)
- [ ] Orchestrate agents via LangGraph or a hand-rolled bounded state machine

## 3. API layer
- [ ] FastAPI: `POST /disclosure/analyze` → job id
- [ ] `GET /jobs/{id}` for status
- [ ] `GET /report/{id}` for the FTO report

## 4. MLOps
- [ ] Docker/docker-compose: app + vector DB + optional local embedding server
- [ ] **Airflow**: scheduled incremental ingestion DAG for new patents in the target CPC classes
      (download → clean → chunk → embed → index)
- [ ] **MLflow**: track retrieval/reranking experiments (embedding models, chunking strategies)
      and log report-generation runs
- [ ] **DVC**: version the ingested patent corpus subset and embedding indices

## 5. Evaluation (the differentiator — real ground truth)
- [ ] Build an eval set from patents with known examiner-cited prior art
- [ ] Run the search+comparison pipeline on those disclosures, score **recall@k** against the
      actual examiner citations — a rigorous, defensible eval design, not just an LLM-judge score

## 6. Tracing
- [ ] OpenTelemetry/Langfuse spans per agent step (search → parse → compare → verify)
- [ ] Explicitly trace the citation-verification guard as a checked step

## 7. Testing
- [ ] pytest for the claims-parser's structured output
- [ ] pytest for the citation-verification guard
- [ ] pytest for chunking logic
- [ ] Integration test over a small fixed patent set with a mocked LLM

## 8. MCP
- [ ] Expose `search-prior-art` as an MCP tool
- [ ] Expose `assess-novelty` as an MCP tool
