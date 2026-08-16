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
