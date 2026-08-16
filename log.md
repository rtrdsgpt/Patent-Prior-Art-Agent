# Build Log — Patent Prior-Art / Freedom-to-Operate Agent

Running log of decisions, reasoning, and failures for this project, kept for interview
preparation. Newest entries at the bottom. Each entry: what happened, why, and what it
cost/taught.

---

## 2026-08-16 — Session start: assessed existing scaffolding

Picked up the project at commit `01daf8e` ("Initial scaffolding: data models, claim-level
chunking, settings, CPC scope decision"). Reviewed what existed before writing anything new:

- `src/patent_agent/schema.py` — Pydantic models (`Patent`, `Claim`, `InventionDisclosure`,
  `SearchResult`, `ClaimElementComparison`, `NoveltyAssessment`, `FTOReport`).
- `src/patent_agent/ingestion/chunking.py` — claim-level chunking with 10 passing unit tests.
- `src/patent_agent/config/settings.py` — env-driven settings (Groq + BigQuery + retrieval
  knobs), CPC scope `G06N3`.
- `docs/cpc_scope.md` — already documents the decision to use **Google Patents Public Data
  via BigQuery** instead of the live USPTO PatentsView API named in `todo.md` section 0. This
  was a prior-session decision, not something I made today: the reasoning recorded there is
  that Google Patents Public Data already contains USPTO's examiner-cited references
  (`citation.category = "EXA"`) structured and SQL-queryable, which is exactly what section 5's
  recall@k eval needs, without the (now-authenticated-only) USPTO Open Data Portal's
  account-creation friction. Confirmed this is still the intended data source and proceeded
  on that basis rather than re-deciding it.

**Decision: pause on anything requiring live credentials.** No `.env` exists in the repo,
and no GCP project / Groq API key were available at session start. Rather than write
ingestion code against BigQuery or agent code against Groq that I could not actually execute
or validate, I asked the user how to handle this. They chose to pause on both the real
BigQuery ingestion client and the Groq-backed agents until credentials are available, but to
keep building everything else against fixture data in the meantime (retrieval stack on local
models, FastAPI skeleton, Docker, test scaffolding, MCP stubs).

**Why this matters for the build order:** todo.md's own ordering ("build the RAG index
before the agents") already implies retrieval should come first, but the credentials gap
pushes this further — the corpus itself has to be a hand-built fixture set until BigQuery
access exists, since section 1 ("download/ingest a corpus subset") is one of the blocked
items. Fixture data lets the whole retrieval stack (embedding index, BM25, hybrid combine,
reranker) be built and unit-tested honestly now, and swapped for a real ingested corpus
later without changing the retrieval code's interface.

## 2026-08-16 — Fixed: package was not pip-installable, `pytest` couldn't collect tests

Ran the existing test suite before changing anything (baseline check). It failed at
collection: `ModuleNotFoundError: No module named 'patent_agent'`. `pyproject.toml` had only
`[tool.pytest.ini_options]` / `[tool.ruff]` sections — no `[build-system]` / `[project]`
table, so nothing had ever installed `src/patent_agent` onto the path. The prior session's
10 chunking tests could only have been run with `src` manually added to `PYTHONPATH`, not via
a normal `pytest` invocation.

**Fix:** added `[build-system]` (setuptools) and `[project]` + `[tool.setuptools.packages.find]`
(`where = ["src"]`) to `pyproject.toml`, matching the sibling `grounded-evals` package's
pattern, then `pip install -e . --no-deps`. `pytest` now collects and passes all 10 existing
tests. Logging this because it's a real gap in what was "done" in the initial commit — the
tests existed but were not actually runnable by a fresh clone + `pip install -r
requirements-dev.txt` + `pytest`, which is what the README's own Setup section promises.
