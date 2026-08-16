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

## 2026-08-16 — Mid-session: user asked me to copy a Groq key from a sibling project's `.env`

The user asked whether I could copy Groq API keys out of the `.env` files of other projects
in the parent `Projects/` directory. The sandbox's auto-mode classifier blocked the shell
command that would have grepped those sibling `.env` files, even with the key values masked
in the output. I did not attempt a workaround (e.g. reading the files a different way) —
the classifier exists specifically to stop an agent from reaching into unrelated projects'
secrets on its own initiative, and "the user asked for it mid-turn" isn't the same as a
durable, explicit permission grant for that class of action. Explained this to the user and
asked them to either paste the key directly or explicitly name the sibling project/loosen
the Bash permission, and continued with the non-blocked work in the meantime. No credentials
were available by the end of this entry, so BigQuery ingestion and the Groq-backed agents
remain paused as decided above.

## 2026-08-16 — Built the fixture corpus + loader (stand-in for BigQuery ingestion)

With real BigQuery ingestion blocked on credentials, hand-built `tests/fixtures/sample_patents.json`:
8 patents in the `G06N3` neighborhood (dropout, CNN+augmentation, LSTM sequence modeling,
attention/seq2seq, batch normalization, GAN, pruning/compression, federated learning), each
with realistic multi-claim text and a mix of `EXA`/`APP` citations between them so the corpus
can later exercise both hybrid retrieval (semantically related but lexically different patents,
e.g. the GAN patent citing the batch-norm patent) and the recall@k eval harness (section 5)
once real data replaces it.

Added `src/patent_agent/ingestion/fixtures.py` (`load_fixture_patents()`) rather than reading
the JSON ad hoc from each retrieval module. Deliberately returns the same `list[Patent]` shape
the real `bigquery_client.py` will return, so retrieval/indexing code written against this
fixture loader doesn't need to change when live BigQuery access lands — only the call site
that produces `list[Patent]` changes, not what anything does with them. This is the same
reasoning as the CPC-scope doc's point about not letting infrastructure gaps leak into the
pipeline's interfaces.

6 new tests in `tests/test_fixtures.py` (claim parsing per patent, independent-claim
detection, `EXA` vs `APP` citation-category filtering via `Patent.examiner_cited_patent_ids`,
and that the loader is cached). Full suite: 16 passed.

## 2026-08-16 — Dense (embedding) retrieval over Chroma

Added `src/patent_agent/retrieval/embedding_index.py`: embeds every claim chunk
(`chunking.claim_to_index_chunk`) with a local `sentence-transformers` model and indexes it
in Chroma, then `dense_search()` queries and collapses results to one `SearchResult` per
patent (best-scoring claim wins) — search operates at the patent level even though the
index is claim-level, since which specific claim tipped off a match is the comparison
agent's job, not the search agent's.

**Design call: explicit embedding-function wrapper instead of Chroma's default.** Chroma
will happily embed for you with a built-in default function, but that default isn't
necessarily the model named in `Settings.embedding_model` — wrapping `SentenceTransformer`
explicitly as a `chromadb.EmbeddingFunction` keeps the actual embedding model a config
decision (`settings.py`), not whatever Chroma ships with, and keeps ingestion/retrieval
consistent if the model choice changes later.

**Failure encountered: two `DeprecationWarning`s from Chroma** (`name()` and `get_config()`/
`build_from_config()` not implemented on the custom embedding function) — Chroma's
`EmbeddingFunction` base class expects these to be overridden for its serialization/registry
system, and warns instead of erroring for now, but says this becomes required in a future
version. Fixed by implementing all three rather than leaving the warnings, since silencing
now and hitting a hard break on a routine `chromadb` bump later is a worse trade than four
extra lines today.

**Test marker decision:** the existing `integration` pytest marker was documented as
"requires live BigQuery/vector-store access (real GCP credentials configured)" — but this
test needs neither GCP nor BigQuery, just a local model download and an in-memory Chroma
client. Reusing `integration` for it would have been misleading (a reader would assume it
needs credentials it doesn't). Added a separate `slow` marker instead: no credentials
needed, just not fast enough to be a default unit test. `tests/test_embedding_index.py`
(5 tests, real local model, no mocking) is marked `slow`; run explicitly with `pytest -m slow`.
Full suite: 21 passed.

## 2026-08-16 — Sparse (BM25) retrieval, and two bugs it surfaced

Added `src/patent_agent/retrieval/bm25_index.py` using `rank_bm25.BM25Okapi`, same
per-patent dedup contract as dense search (best-scoring claim wins) so the two result lists
are directly combinable in the hybrid step. Simple regex tokenizer (`[a-z0-9]+` on
lowercased text), applied identically at index and query time.

**Failure 1 — `ZeroDivisionError` on an empty corpus.** `build_bm25_index([])` crashed
inside `rank_bm25`'s own `__init__` (`self.avgdl = num_doc / self.corpus_size`, dividing by
zero) rather than raising something meaningful. Not a hypothetical case — `bm25_search`
needs to behave sanely before any real corpus is ingested, and an integration test wants to
assert exactly this "empty index, empty results" behavior. Fixed by making `BM25Index.model`
`Optional`, skipping the `BM25Okapi(...)` construction entirely when there are no chunks, and
short-circuiting `bm25_search` to `[]` when `model is None`, rather than trying to work
around `rank_bm25`'s internals.

**Failure 2 — bad test fixture, not a code bug.** Wrote a "nonsense query returns no
results" test using the query `"xyzzy nonexistent term plugh"`, expecting zero matches. It
failed: `term` turned out to be a real token in the corpus, because the tokenizer splits on
non-alphanumeric characters and "short-term" (from the LSTM patent's "long short-term
memory") tokenizes to `short` + `term`. This is correct tokenizer behavior, not a bug — the
test's assumption was wrong, not the code. Fixed the test query to use words with no
substring overlap with anything in the fixture corpus (`"zzyzx qwibble florp vandelay"`).
Worth remembering for later: BM25 term matching operates on tokens, so hyphenated compound
terms can produce surprising partial matches — relevant if real patent claims contain a lot
of hyphenated technical terminology (they do).

7 new tests in `tests/test_bm25_index.py` (exact-terminology ranking, dedup, descending
scores, retrieval_method tag, zero-score exclusion, top_k, empty index). Not marked `slow` —
BM25 is pure Python/numpy over pre-tokenized text, no model download. Full suite (excluding
`slow`): 23 passed.

## 2026-08-16 — Hybrid retrieval via Reciprocal Rank Fusion

Added `src/patent_agent/retrieval/hybrid.py`, closing out todo.md section 1.

**Decision: RRF over raw score combination.** BM25 scores and Chroma's distance-derived
dense scores are on different, uncalibrated scales — BM25 scores are unbounded and corpus-
size-dependent, the dense score here is a `1/(1+distance)` transform with its own range.
Summing or weighting them directly would mean whichever ranker happens to produce larger
raw numbers dominates the fused ranking, which isn't a real judgment about relevance, just
an artifact of two different scoring functions' scales. RRF (`score = sum(1/(rrf_k + rank))`
per ranker, `rrf_k=60` — the standard literature default) sidesteps this by only using each
ranker's *rank order*, not its raw scores, which is why it's the standard choice for
combining heterogeneous rankers rather than something bespoke here.

`hybrid_search()` deliberately over-fetches each ranker to its own `bm25_top_k`/
`dense_top_k` (already in `Settings` from the initial scaffolding) before fusing down to
`hybrid_top_k` — fusing from a wider pool than the final cut is what lets a patent only one
ranker ranked highly still surface in the fused list, instead of only rewarding consensus.

13 new tests: 6 fast unit tests against `reciprocal_rank_fusion` directly (using synthetic
`SearchResult` lists, no real indexes — verifying the fusion math itself: consensus ranks
first, single-ranker results are still included, descending scores, `top_k` respected,
retrieval_method tagging, empty input), plus 1 `slow`-marked end-to-end test building real
BM25 + embedding indexes over the fixture corpus and checking the fused top result matches
what both rankers should agree on. Full suite: 35 passed.

*Correction to that commit message:* it said "closes todo.md section 1" — that was
premature, the section also lists cross-encoder reranking, which wasn't done yet (see next
entry). Noting the correction here rather than rewriting the pushed commit.

## 2026-08-16 — Cross-encoder reranking, closing out todo.md section 1

Added `src/patent_agent/retrieval/reranker.py` using `sentence-transformers`'
`CrossEncoder` (`cross-encoder/ms-marco-MiniLM-L-6-v2`, already named in `Settings` from the
initial scaffolding). This actually finishes todo.md section 1 (the previous commit's claim
to have closed it was wrong — see correction above).

**Why a reranking stage at all, given hybrid retrieval already ranks candidates:** BM25 and
dense retrieval both score a query against a document independently and then compare
vectors/scores (bi-encoder style) — necessary to scale to a full corpus, but structurally
unable to model interaction between the specific query and a specific candidate's text. A
cross-encoder feeds the (query, candidate) pair through the model jointly, which is far more
accurate at judging true relevance but too slow to run over an entire corpus. So it only
runs over the hybrid stage's already-narrowed candidate list (`hybrid_top_k`, currently 20) —
narrowing a lot before the expensive step, not instead of it.

**Design call: rerank against abstract + independent claims, not a single claim.** Hybrid
retrieval's `SearchResult` is already deduplicated to one score per patent (see the dense/
BM25 entries above for why), so by the time reranking runs there's no single "the chunk that
matched" to rerank against — reconstructing that would mean threading per-claim provenance
through the hybrid/RRF step just for this. Instead, reranking scores the query against each
candidate patent's abstract plus its independent claims joined — independent claims because
those define the actual scope of what's protected (dependent claims narrow an already-
independent claim, so they add detail but not new scope), and the abstract because it's
almost always the most on-topic single paragraph in the document. This is a real trade-off
(a highly specific dependent claim can occasionally be the true match) worth remembering: if
reranking precision on real data turns out weak, the fix is threading claim-level
provenance through hybrid retrieval, not just swapping the reranker model.

6 new `slow`-marked tests (real cross-encoder, no mocking): true-positive-over-distractors
ranking (the actual point of reranking — does it fix a case dense/BM25 alone might not rank
first), `rerank_top_k` respected, retrieval_method tagging, descending scores, candidates
missing from the patent lookup are skipped rather than erroring, empty candidates. Full
suite: 41 passed, ~98s (dominated by loading the embedding + cross-encoder models across the
`slow` tests — acceptable for now since nothing here is on a hot path yet, worth revisiting
if the `slow` suite becomes a CI bottleneck later).

## 2026-08-16 — FastAPI layer (todo.md section 3), honest about what's not built yet

Added `src/patent_agent/api/` (`jobs.py`, `pipeline.py`, `app.py`): `POST
/disclosure/analyze` → job id, `GET /jobs/{id}` → status + candidates, `GET /report/{id}` →
FTO report.

**Key decision: the report route returns 501 with the candidate list, not a fabricated
report.** Only retrieval (section 1) is built; the disclosure-parser/comparison/risk-report
agents (section 2) are paused pending the Groq key. A tempting shortcut here would be to
stub those agents with hardcoded or trivial logic so `/report/{id}` "works" end-to-end for a
demo. Deliberately didn't — a report-shaped response with no real novelty assessment behind
it is worse than an honest 501, because it would look done and isn't; the whole point of
this project (per todo.md/README) is that every claim in the report is grounded and
verified, so a fake-grounded report is actively the wrong direction to cut a corner in.
`/jobs/{id}` does return real candidate patents, though — that part is genuinely working
(hybrid search + rerank over the corpus), just not the whole pipeline.

**`run_prior_art_search()` uses the disclosure's raw text as the retrieval query directly**,
not a structured extraction of it — because the disclosure-parser agent that would produce
`technical_field`/`key_elements`/`candidate_cpc_classes` doesn't exist yet. Once it does,
only the query-construction step changes; the retrieval call itself is already correct.

**Design call: in-memory job store, not a database.** No multi-worker deployment exists yet,
so persistence would be speculative. `JobStore` is thread-safe (a `Lock` around the dict)
because FastAPI runs sync route handlers in a thread pool — concurrent requests are real
even single-process. Indexes are built once per process via `lru_cache` on `_get_indexes()`,
not per-request, since embedding a corpus is real work.

18 new tests: `test_jobs.py` (6, pure `JobStore` logic) and `test_api.py` (6, via
`TestClient`, with `run_prior_art_search` monkeypatched out — these test the API's
contract/status codes, not retrieval accuracy, which is already covered by
`test_hybrid.py`/`test_reranker.py`). Noticed but didn't chase: `starlette.testclient`
warns that its `httpx`-based `TestClient` is deprecated in favor of a package called
`httpx2` — real upstream package, but a test-infra-only concern with no production impact
right now; not worth a new dependency for this yet. Full suite: 53 passed.

## 2026-08-16 — Mid-session: user set up GCP (project, billing) for BigQuery access

User installed `gcloud` (to `~/Downloads/google-cloud-sdk`, not on `PATH` by default —
needed `export PATH="$HOME/Downloads/google-cloud-sdk/bin:$PATH"` to invoke it) and created
`patent-prior-art-project`. Checked `gcloud services list --enabled` — BigQuery API was
already enabled — but `gcloud billing projects describe patent-prior-art-project` showed
`billingEnabled: false` and `gcloud billing accounts list` returned zero accounts. BigQuery
requires billing enabled on the querying project even for free-tier usage against public
datasets like Google Patents Public Data (query costs bill to the querying project, not the
dataset's own project), so this was a real blocker, not a formality. Asked the user how they
wanted to handle it rather than trying to script billing-account creation myself — linking a
real billing account is exactly the kind of action-with-financial-consequence that should be
the user's explicit, deliberate action, not something run on their behalf. They set it up
themselves; re-checked afterward and confirmed `billingEnabled: true`.

Still outstanding: Application Default Credentials (`gcloud auth application-default login`)
— this needs an interactive browser flow, so it can't be run from here; gave the user the
exact command to run in their own terminal. Once that's done, the real BigQuery ingestion
client (`ingestion/bigquery_client.py`, section 0/1 of todo.md) can finally be built and
actually tested against Google Patents Public Data, not just written and left unexecuted.

## 2026-08-16 — Chroma client now configurable for docker-compose

Before building `docker-compose.yml`, made the app layer actually able to talk to a Chroma
*service* instead of only ever embedding Chroma in-process. Added `chroma_host`/`chroma_port`
to `Settings` (both optional, default unset) and `_build_chroma_client()` in
`api/pipeline.py`: uses `chromadb.HttpClient` when `chroma_host` is set (docker-compose sets
it via env var to reach the sibling `chroma` container), otherwise falls back to
`chromadb.PersistentClient` writing to `settings.chroma_persist_directory` — so a local
(non-Docker) run of the API doesn't silently use an in-memory ephemeral store that forgets
the corpus every restart, the way `embedding_index.build_embedding_index`'s own default
still does for its unit tests.

**Testing note:** `chromadb.HttpClient(...)` connects eagerly at construction time and
raises `ValueError` immediately if nothing is listening — confirmed this experimentally
before writing the test, rather than assuming lazy connection. So
`test_build_chroma_client_uses_http_client_when_host_set` monkeypatches
`chromadb.HttpClient` itself (asserting it's called with the right host/port) instead of
spinning up a real Chroma server just to test a branch of config logic.

4 new tests in `tests/test_pipeline.py`. Full suite: 55 passed (~110s, dominated by the
`slow` model-loading tests as before).

## 2026-08-16 — Credentials unblocked: real BigQuery ingestion client

User completed both remaining GCP steps: `gcloud auth application-default login`
(interactive browser flow, run in their own terminal per my instructions) and confirmed
billing was enabled. Verified independently before writing any ingestion code:

- `gcloud auth application-default print-access-token` — succeeded, ADC live.
- `gcloud billing projects describe patent-prior-art-project` — `billingEnabled: true`.
- Ran a real `bq query` against `patents-public-data.patents.publications` filtered to
  `G06N3%` CPC codes — got back real patents (e.g. "Synapse element and neuromorphic
  processor including synapse element"). This is the point where BigQuery ingestion stopped
  being paused and became something I could actually build and test, not just write.
- Checked the live table schema with `bq show --schema` *before* writing the ingestion
  query, rather than assuming the field names from memory/docs. Confirmed
  `title_localized`/`abstract_localized`/`claims_localized` are `RECORD REPEATED` with
  `text`/`language` subfields (need `language = 'en'` filtering + `UNNEST`), `cpc` is
  `RECORD REPEATED` with a `code` subfield (my first hand-typed test query got this wrong —
  tried `ARRAY_LENGTH(cpc.code)` as if `cpc.code` were a flat array field, got "Cannot
  access field code on a value with type ARRAY<STRUCT<...>>"; fixed by using
  `EXISTS(SELECT 1 FROM UNNEST(cpc) AS c WHERE c.code LIKE ...)`), and `citation.category`
  values match `CitationCategory`'s `EXA`/`APP` exactly as the initial-scaffolding schema
  assumed.

Added `src/patent_agent/ingestion/bigquery_client.py` (`fetch_patents()`), following the
same `list[Patent]` return contract as `ingestion/fixtures.py` (see that module's log entry
for why that contract was chosen up front). Design notes:

- **`citation.category` maps to `OTHER` for anything that isn't `EXA`/`APP`.** The real
  field has 11 possible values (search-report types, opposition, appeal, etc. — see the
  field's own BigQuery description); `CitationCategory` only distinguishes examiner vs.
  applicant citations because that's the only distinction the recall@k eval (section 5)
  actually needs. Collapsing the rest to `OTHER` rather than modeling all 11 is deliberate
  scope discipline, not laziness — nothing downstream needs finer granularity yet.
- **Rows with no parseable claims are dropped during ingestion, not carried through as
  patents with an empty claims list.** A patent this pipeline can't chunk into claims can't
  be compared against a disclosure element-by-element later, so keeping it around would
  just be dead weight that every downstream consumer has to defensively check for.
- **`publication_date` malformed/zero values become `None`, not a raised error** — ingesting
  300 patents at a time, a handful having a bad or missing date shouldn't fail the whole
  batch over a field nothing in the pipeline treats as required.
- Added `Settings.corpus_size` (default 300) — a new explicit cap distinct from the
  retrieval-stage `*_top_k` settings, keeping BigQuery cost/time bounded per the "scope
  discipline" already written into `docs/cpc_scope.md`.

Created the real (gitignored) `.env` with `GCP_PROJECT_ID=patent-prior-art-project` —
`GOOGLE_APPLICATION_CREDENTIALS` deliberately left blank since ADC (not a service-account
key file) is what's configured; `.env.example`'s own comment already said ADC works in
place of a key file.

10 new tests in `tests/test_bigquery_client.py`: 9 fast unit tests against `_row_to_patent`/
`_parse_publication_date` using hand-built `google.cloud.bigquery.table.Row` objects (no
network) covering date parsing edge cases, missing-claims rows, citation category mapping,
missing title/abstract; plus 1 `integration`-marked test that runs the real query against
live BigQuery (skips itself if `GCP_PROJECT_ID` isn't configured, so the suite still runs
clean for anyone without these credentials) and asserts the results are genuinely CPC-scoped
US patents with parsed claims. Full suite: 65 passed.

## 2026-08-16 — Docker/docker-compose (todo.md section 4, partial), and a real bug it caught

Added `Dockerfile`, `docker-compose.yml`, `.dockerignore` — app + Chroma vector-DB service,
the two pieces of section 4 that don't depend on paused credentials (Airflow's scheduled
ingestion DAG and MLflow experiment tracking still need a real, repeated ingestion/retrieval
workflow to actually track, so those stay deferred; see next entries for why they weren't
just stubbed out anyway). Skipped the "optional local embedding server" mentioned in
todo.md's phrasing — embedding already runs in-process via `sentence-transformers`
(`embedding_index.py`), so a dedicated embedding server would be infrastructure with no
current consumer, not a real requirement.

**Dependency the Dockerfile needed that wasn't obvious until building it:**
`ingestion/fixtures.py` resolves its corpus path relative to the repo root
(`Path(__file__).resolve().parents[3] / "tests" / "fixtures"`), so the image needs
`tests/fixtures/` copied in even though `tests/` is otherwise dev-only. Noting this as a
known interim wart, not fixing it now: once real BigQuery ingestion is what the running app
actually uses (see below — that's not wired up yet either), this dependency goes away on its
own rather than needing a deliberate fix.

**Did not stop at "docker build succeeded" — actually ran `docker compose up` and drove the
API through it**, since a container that builds isn't the same as a container that works
(model downloads at import time, network access to HuggingFace from inside the container,
Chroma's own container startup, and `chroma_host`-based `HttpClient` wiring were all real
things that could have failed at runtime even with a clean build). Posted a real disclosure
("dropout during training") to `POST /disclosure/analyze` through the compose network,
polled `/jobs/{id}` to `completed`, and confirmed the dropout patent (`US10000001B2`) ranked
first — the same result the `slow` reranker test already established, but now verified
through the actual deployed path, not just direct Python calls. Also confirmed `/report/{id}`
returns the honest 501 with the candidate list through the real container.

**Real bug found by this end-to-end run, not by unit tests:** `/report/{id}`'s 409 "job not
finished yet" message rendered as `"status: JobStatus.RUNNING"` instead of `"status:
running"` — `f"{job.status}"` on a `class JobStatus(str, Enum)` prints the enum's
`repr`-style name in this Python version, not its string value, despite the class mixing in
`str`. None of the unit tests caught this because none of them asserted on that particular
error message's exact text; it only became visible reading real `curl` output against the
running container. Fixed by using `job.status.value` explicitly in the f-string. Worth
remembering generally: `str`-mixin enums don't reliably `str()`/format to their value across
Python versions — call `.value` explicitly wherever the string form leaves the class.
Rebuilt the image and reran the same manual verification (including the `run_prior_art_search`-mocked GAN-patent case, to confirm a different retrieval ranking hits the same path
correctly) after the fix.

## 2026-08-16 — MCP tool exposure (todo.md section 8)

Added `src/patent_agent/mcp_server.py` using the `mcp` package's `MCPServer` (a FastMCP-style
API — this SDK version, 2.0.0, has it at `mcp.server.mcpserver.MCPServer` rather than the
`mcp.server.fastmcp.FastMCP` path documented in older MCP SDK material, so I confirmed the
actual class location, constructor, and `@mcp.tool()`/`call_tool()` signatures via `inspect`
against the installed package rather than assuming from memory before writing anything).

Two tools, same honesty principle as the `/report` API route: **`search_prior_art`** is real
— it calls the same `run_prior_art_search()` the FastAPI layer uses, so it's genuinely
backed by hybrid retrieval + reranking over the indexed corpus. **`assess_novelty`** raises
`mcp.server.mcpserver.exceptions.ToolError` with a message naming exactly what it depends on
(the paused comparison agent) rather than returning fabricated or empty novelty data.

**Verified the error actually surfaces as an error, not a silently-swallowed one:**
experimentally called `mcp.call_tool("assess_novelty", ...)` directly before writing the
test and confirmed the `ToolError` propagates out as a real raised exception at this API
level (the SDK's own `tool.run()` wraps and re-raises it) — so
`test_assess_novelty_raises_not_implemented_tool_error` asserts `pytest.raises(ToolError,
match=...)` rather than checking a result object's `is_error` flag, which is what
`search_prior_art`'s success-path tests check instead (`CallToolResult.structured_content`).

4 new tests in `tests/test_mcp_server.py`, using `pytest.mark.anyio` (the `anyio` pytest
plugin ships with the `anyio` package already in the dependency tree — no separate test
dependency needed) since `MCPServer.call_tool`/`list_tools` are async. Added `mcp>=2.0.0` to
`requirements.txt`. Full suite: 69 passed.

## 2026-08-16 — User-requested restructure: flatten `src/patent_agent/` to `src/`

User asked, mid-session, to drop the `patent_agent` package folder and have modules live
directly under `src/` (`src/schema.py`, `src/ingestion/`, `src/api/`, etc., rather than
`src/patent_agent/schema.py` etc.). Confirmed the exact target shape before touching
anything, since "move the files" was ambiguous about whether `src/` itself should also go
away — user confirmed: keep the `src/` layout, just remove the `patent_agent` nesting level.

Mechanical but repo-wide: `git mv` for every module/package (preserves history rather than
delete+recreate), then a blanket `sed 's/patent_agent\.//g'` across every `.py` file in
`src/` and `tests/` to fix ~61 import lines and monkeypatch target strings (e.g.
`"patent_agent.mcp_server.run_prior_art_search"` → `"mcp_server.run_prior_art_search"`) in
one pass, followed by hand fixes for what a blind string replace couldn't catch:

- `ingestion/fixtures.py`'s repo-root path resolution (`Path(__file__).resolve().parents[N]`)
  needed `N` decremented by one — it counts directory levels up to the repo root, and there's
  now one fewer level between the module and the root.
- `pyproject.toml`: `[tool.setuptools.packages.find]` alone doesn't discover standalone
  modules (`schema.py`, `mcp_server.py` have no `__init__.py`/parent package now, they're
  just files directly under `src/`) — added `package-dir = {"" = "src"}` and `py-modules =
  ["schema", "mcp_server"]` explicitly.
- `Dockerfile`'s `CMD` referenced `patent_agent.api.app:app`.
- `README.md`/`docs/cpc_scope.md`/`.env.example` had hardcoded `src/patent_agent/...` paths
  in prose, not code the sed pass would touch.

**Verification, not just "tests still pass":** reinstalled the package
(`pip install -e . --no-deps`), confirmed every module imports cleanly, ran the full test
suite (69 passed) and `ruff check` (clean), then — since a Python-level pass doesn't prove
the *deployed* path works — ran the API locally with `uvicorn api.app:app` and drove a real
`/disclosure/analyze` → poll `/jobs/{id}` → `completed` cycle by hand (LSTM patent
`US10000003B2` correctly ranked first for an LSTM-flavored disclosure), the same kind of
real check used for the Docker deployment earlier. No new bugs surfaced from the restructure
itself, unlike the Docker one.

## 2026-08-16 — Wired the real ingested corpus into the pipeline — and found the CPC-scope
doc's core evaluability claim was wrong

Added `ingestion/corpus.py` (`load_corpus()`/`save_corpus()`) and
`ingestion/ingest_corpus.py` (`python -m ingestion.ingest_corpus`, a one-shot CLI, not yet
the Airflow DAG from todo.md section 4 — see the Docker entry above for why that's deferred).
`load_corpus()` reads a cached `data/corpus.json` if one exists, else falls back to the
fixture set — same fallback reasoning as `fixtures.py` itself. `api/pipeline.py`'s
`_get_indexes()` now calls `load_corpus()` instead of `load_fixture_patents()` directly, so
the API/MCP tools search whatever corpus is actually cached, real or fixture, without caring
which. `data/` added to `.gitignore` — this is real bulk data pulled from BigQuery, not
source, and properly versioning it is todo.md section 4's DVC item, not a plain git commit.

Ran `python -m ingestion.ingest_corpus` for the first time against live BigQuery: 291
patents ingested. Then, out of habit of checking a new dataset rather than trusting it,
inspected the result — `examiner_cited_patent_ids` was empty for **all 291 patents**. That's
not a plausible outcome for a targeted CPC slice this size, and it directly contradicts
`docs/cpc_scope.md`'s stated reason for picking `G06N3` in the first place ("real ground
truth exists... `citation.category = 'EXA'`... this was checked before picking the class,
not assumed").

**Investigated with live `bq query` rather than guessing:**
- Grouped `citation.category` by value across the *entire* live `patents.publications`
  table (not just the 291-patent sample) — zero rows anywhere contain `"EXA"`, across
  ~3.6M citation rows in the G06N3-scoped slice alone. Whatever check the CPC-scope doc's
  author believed they'd done, this value is not actually populated in the live data today.
- Grouped again to see what values *are* populated: `APP` (1.89M), `SEA` (908K+),
  `PRS`/`ISR`/etc. in smaller numbers — and, importantly, the raw field isn't single-valued:
  real rows contain comma-joined compound strings like `"APP,APP"` or `"PRS,SEA"`, which an
  exact-match dict lookup (the original `_CATEGORY_MAP`) would silently miss entirely,
  compounding the problem.
- Reasoned about *why* `SEA` (not `EXA`) is what's populated: Google Patents Public Data
  sources its `citation` field from EPO/DOCDB's harmonized citation categories, and DOCDB's
  own taxonomy uses `SEA` ("search report") for citations added during the patent office's
  own prior-art search — which for a US application *is* the examiner's search. `EXA` is
  listed in the field's own BigQuery description as if it were a live category, but isn't
  actually one DOCDB harmonization populates for this table. Confirmed `SEA`-category rows
  carry `type` values `X`/`Y` (DOCDB's own "particularly relevant" relevance codes), which is
  the concrete signal that these are genuinely examiner-flagged prior art, not incidental.

**Fix:** `bigquery_client.py`'s category mapping is now `_map_citation_category()` — splits
the raw string on comma and checks token membership (`SEA` → `CitationCategory.EXAMINER`,
`APP` → `CitationCategory.APPLICANT`, `SEA` takes precedence when both appear together, e.g.
`"APP,SEA"`, since being flagged in the office's own search is the stronger signal). Our own
`CitationCategory.EXAMINER` enum member keeps its existing internal value (`"EXA"`) —
that's our domain vocabulary, not required to match Google's raw source string, and changing
it would have broken the already-correct hand-authored fixture corpus
(`tests/fixtures/sample_patents.json`, which encodes `"EXA"` directly since it's clean data
authored by hand, not translated from a raw BigQuery code). Re-ran ingestion after the fix:
**149 of 291 patents** now have at least one examiner-cited reference — a real, usable
ground-truth signal for the section 5 recall@k eval work.

Added a `pytest.mark.integration` regression test,
`test_fetch_patents_live_corpus_has_examiner_citations`, specifically so this exact failure
mode (category mapping silently matching nothing) can't regress unnoticed again — it asserts
a live 50-patent sample actually yields examiner citations, not just that ingestion runs
without erroring. Also added 3 fast unit tests for the compound-value splitting and
`SEA`-precedence behavior, and updated the existing category-mapping test to use `"SEA"`
instead of the now-known-fictional `"EXA"`. Full suite: 77 passed.

**Why this is worth dwelling on:** this is exactly the kind of assumption that looks
verified in a planning doc (a specific field name, a specific value, cited as "checked, not
assumed") but wasn't actually checked against live data — it was checked against the field's
*documentation*, which turned out to describe a broader taxonomy than what's actually
populated. The fix was cheap once found (one BigQuery grouping query, then a small mapping
change), but finding it required actually inspecting ingested output rather than treating
"ingestion ran without an error" as "ingestion is correct."

## 2026-08-16 — Groq credentials landed: rotating multi-key client

User pasted a `GROQ_API_KEY` value into `.env` that turned out to be five comma-separated
keys, not one — consistent with their earlier ask (mid-session) about copying "the groq keys
(multiple)" from other projects. Rather than silently use only the first and leave four idle,
asked how they wanted the multi-key situation handled; they chose key rotation on rate-limit
over "just use the first."

**Why rotation is a real requirement here, not gold-plating:** Groq's free tier has fairly
tight per-key rate limits, and todo.md's agent pipeline (section 2) will make several LLM
calls per job across multiple stages (disclosure-parser → comparison → risk-report, at
minimum) — a single free-tier key would throttle a multi-agent pipeline quickly once real
usage starts, and the user specifically has five keys available for exactly this reason.

Added `Settings.groq_api_keys` (a property parsing the comma-separated `groq_api_key` field —
kept the single-string field name so `.env`'s `GROQ_API_KEY=` stays one plain env var, not a
JSON-array env var pydantic-settings would otherwise expect) and
`agents/groq_client.py`'s `RotatingGroqClient`: wraps one `groq.Groq` client per key, catches
`groq.RateLimitError` specifically (not a bare `except Exception`, since only a 429 means
"try a different key" — any other error is a real failure that rotating keys wouldn't fix),
and rotates to the next key, bounded to one attempt per configured key (same bounded-retry
discipline as todo.md's note about the search agent's query-expansion retries). Rotation
state (`_current`) is sticky across calls, not reset per call, so once an earlier key is
exhausted, later calls start from wherever rotation left off instead of re-hitting it.

**Verified against the live Groq API, not just the mocked rotation logic:** called
`models.list()` first to see what's actually available rather than guessing a model name —
picked `llama-3.3-70b-versatile`, a well-established model with reliable JSON-mode support,
over newer options in the list (`openai/gpt-oss-120b`, `qwen/qwen3.6-27b`) that I have no
track record with for structured-extraction tasks, which is what the disclosure-parser/
claims-parser agents will need. Then ran one real `chat_completion()` call through
`RotatingGroqClient` end-to-end and confirmed a real response came back — same "don't trust
it until it's exercised for real" standard applied to BigQuery and Docker earlier in this
session.

6 unit tests (`tests/test_groq_client.py`, `Groq` class monkeypatched, a hand-built
`httpx.Response`/`RateLimitError` to simulate a 429 without a network call) covering: single
key with no rate limit, rotation on rate-limit, exhausting all keys raises
`NoAvailableGroqKeyError`, rotation state stays sticky across separate calls, empty key list
rejected at construction, `build_groq_client()` reads `Settings` correctly. Plus 1
`integration`-marked live test. Full suite: 84 passed.

Credentials are now fully unblocked (BigQuery *and* Groq) — the next real todo.md work is
section 2's actual agents (disclosure-parser, search, claims-parser, comparison,
risk-report), which is the biggest remaining chunk of the project.

## 2026-08-16 — Disclosure-parser agent: the first real LLM pipeline stage

Added `agents/disclosure_parser.py` (`parse_disclosure()`): free-text invention disclosure
→ populated `InventionDisclosure` (`technical_field`, `key_elements`,
`candidate_cpc_classes`) via Groq JSON mode. `schema.py`'s `InventionDisclosure` already had
these three fields sitting unset since the initial scaffolding commit, with a docstring
pointing at `agents/disclosure_parser.py` — this is the first agent that actually fills them.

**JSON mode is not schema validation — confirmed this distinction matters, didn't just
assume it.** Groq's `response_format={"type": "json_object"}` guarantees syntactically valid
JSON, not that it has the three keys this pipeline needs in the right shapes (e.g. the model
could return `key_elements` as a string instead of an array). So `parse_disclosure()` still
runs the parsed JSON through `InventionDisclosure(...)` construction and catches
`ValidationError`/`KeyError`/`TypeError`/`JSONDecodeError` as one bounded-retry class: on
failure, the bad response and a correction request go back into the message history and it
tries once more (`_MAX_ATTEMPTS = 2`), then raises clearly rather than looping — same
bounded-retry discipline as the Groq client's key rotation and the (still-upcoming) search
agent's query expansion.

**Verified live, both the happy path and that live output is actually sensible, not just
schema-shaped:** ran a real disclosure ("CNN with dropout regularization on fully-connected
layers...") through the live model and asserted the extracted `key_elements` actually
mentions "dropout" — checking that the LLM call succeeds isn't the same as checking it
extracted something *correct*, and a schema-valid-but-wrong response would have passed a
weaker test.

6 tests in `tests/test_disclosure_parser.py`: 5 fast unit tests against a mocked
`RotatingGroqClient` (populated happy path, correct model/JSON-mode passed to the client,
retry-then-succeed on malformed JSON, retry-then-succeed on a missing key, raises after
exhausting retries) plus 1 live `integration` test. Full suite: 90 passed.

## 2026-08-16 — Prior-art search agent, wired into the API, and a real perf bug found

Added `agents/search_agent.py` (`search_prior_art()`): the second section-2 agent. Reuses
`retrieval/hybrid.py` and `retrieval/reranker.py` completely unchanged — this agent's actual
job is query *construction* from the disclosure-parser's structured output, plus the
adaptive-query-expansion retry todo.md calls for, not new retrieval mechanics.

**Adaptive expansion is deterministic, not another LLM call — a deliberate trade-off.**
Considered having the agent call the LLM again to rewrite a too-narrow/too-broad query, which
is the more literal reading of "adaptive query expansion." Went with a fixed 3-query sequence
built from fields the disclosure-parser already extracted instead (medium: technical_field +
all key_elements; broad: technical_field alone; narrow: technical_field + first two
key_elements): it's real and testable without mocking another LLM call, it's free (no extra
Groq call/rate-limit exposure per search), and it reuses data already paid for in the
disclosure-parser step rather than asking the model to reinvent it. Bounded to those 3 fixed
attempts — same discipline as the Groq client's key rotation and the disclosure-parser's
retry loop: try a small fixed set, then stop and return the best attempt.

**"Too few/too many" is judged by reranked score, not result count** — `rerank_top_k`
already caps the returned list at a fixed size, so count alone can't signal over-broad vs.
over-narrow. Cross-encoder scores are logit-style and roughly centered on 0 for relevance
(observed range roughly -12 to +10 across `reranker.py`'s tests), so "how many of the
returned results score above 0" is a real, if heuristic, proxy for "how many look plausibly
relevant." `_MIN_RELEVANT=1`/`_MAX_RELEVANT=8` are starting points, explicitly flagged in
the docstring as unvalidated against ground truth — section 5's recall@k eval is what would
actually justify or adjust these numbers once it exists, not intuition.

Wired `api/pipeline.py`'s `run_prior_art_search()` to actually call `parse_disclosure()` then
`search_prior_art()` instead of using the raw disclosure text as the query — the placeholder
both this module's and `agents/disclosure_parser.py`'s docstrings had been pointing at since
they were written. Updated `api/app.py`'s 501 report message accordingly (it previously said
the comparison/risk-report agents were "paused pending a Groq API key," which stopped being
true this session — now it correctly says they aren't built yet, a different and more
accurate reason).

**Real bug found by actually running the wired-up pipeline end-to-end, not by inspection:**
ran `run_prior_art_search()` against the real 291-patent corpus and noticed repeated
"Loading weights" output — the cross-encoder reranker was reloading its model from scratch
on every single `rerank()` call. Traced it to `reranker.py`'s own default path: `model = model
or CrossEncoder(settings.reranker_model)` constructs fresh every time `model` isn't passed,
and nothing was passing it — the parameter existed (with a docstring claiming callers "can
load the cross-encoder once and reuse it") but no caller actually did. Harmless before this
session (reranking only ran once per test), but now that `search_agent.py`'s adaptive retry
can call `rerank()` up to 3 times per search, and the API calls this per user request, it was
a real, request-scaling cost. Fixed with `_default_model()`, an `lru_cache`-keyed-by-model-
name helper, used as the default when `model` isn't explicitly injected (tests still inject
stubs directly, unaffected). Confirmed the fix with a repeat-call trace: 3 `rerank()` calls
in one process now trigger exactly 1 model load, not 3.

**Investigated a second "Loading weights" source and concluded it wasn't worth fixing.**
`build_embedding_index()` was also printing multiple "Loading weights" lines. Traced it with
instrumented `__call__`/`build_from_config` methods on the embedding function: Chroma calls
`build_from_config()` (the method added earlier this session to silence a deprecation
warning — see that log entry) a few times *during collection setup* — likely internal
config-round-trip validation — but the actual embedding computation (`__call__`) reuses one
consistent instance across both index-build and every subsequent query. So this cost is
one-time per process (paid once when `_get_indexes()` populates its cache), not per-request
like the reranker bug was — not worth adding complexity to avoid a bounded, non-scaling cost.
Noting the investigation and the reasoning for *not* fixing it, since concluding something
isn't worth fixing is still a real decision, not the same as not checking.

10 new tests: 3 for `_candidate_queries`' exact query construction, 5 for
`search_prior_art`'s retry/early-exit/best-attempt logic (hybrid_search/rerank monkeypatched
to control relevance counts directly — this agent's orchestration logic, not retrieval
correctness, is what's under test here), 1 `slow`-marked real end-to-end test over the
fixture corpus. Updated `test_api.py`'s 501-message assertion to match the corrected wording.
Full suite: 98 passed.
