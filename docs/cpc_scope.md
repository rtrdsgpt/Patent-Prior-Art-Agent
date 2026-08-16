# CPC scope for the first working slice

**Chosen class: `G06N3` — "Computer systems based on biological models" (neural networks
and related architectures).**

## Why this class

- **Well-populated but not enormous.** G06N3 has seen heavy filing activity since ~2012
  (the deep-learning patent boom) without being as sprawling as a top-level class like
  `G06N` (all computing-based-on-specific-models) or `G06F` (general computing). Large
  enough to build a real hybrid-retrieval index and a meaningful eval set from; small
  enough that a first working slice (ingest → index → search → compare → verify) doesn't
  turn into an unbounded data-engineering project before the agent pipeline exists.
- **Legally meaningful, technically legible claims.** Neural-network patent claims tend to
  describe concrete architectural/training-method elements (layers, activation functions,
  training procedures) that are easier to element-by-element compare against an invention
  disclosure than, say, chemistry or mechanical claims full of domain-specific notation —
  useful for both building and *demonstrating* the comparison agent's output.
  matches the domain the rest of the CV portfolio (Legal SLM, Financial RAG, DatoScope) is
  already in, without the project itself being finance/legal-themed (per the original scoping
  decision that this should be a fresh "real problem," not a finance-themed system).
- **Real ground truth exists.** Google Patents Public Data's `citation` field gives
  examiner-cited prior art for patents in this class, which is what makes the recall@k
  evaluation (section 5 of `todo.md`) possible at all. **Correction, added once the pipeline
  was actually built and run against live data:** the field's own documentation lists `EXA`
  as the examiner-citation category value, but that value is never actually populated in the
  live table — `SEA` (DOCDB's "search report" category) is what's real. Confirmed with a
  live `bq query` grouping citations by category before fixing the ingestion code; see
  `log.md`'s 2026-08-16 entries for the exact query and reasoning. The underlying claim
  (real ground truth exists for this class) held up — the specific field value cited for it
  originally didn't, and the code now reflects the corrected version, not the original one.

## Scope discipline

Per the portfolio plan, don't expand CPC coverage until the first working slice — one
class, a small indexed corpus, search → compare → verify working end-to-end — is real.
Widening to sibling classes (e.g. `G06N20` — machine learning generally) is a natural
second-slice expansion once the pipeline works, not a day-one decision.

**Update:** the corpus did expand once the pipeline was real enough to need it — not to a
new CPC class, but along citation edges within the existing one. A plain `LIMIT`-based seed
fetch (no `ORDER BY`) turned out to have ~0% chance of containing any given seed patent's own
cited prior art (confirmed empirically, not assumed — see `log.md`), which made the
recall@k eval's "in-corpus" metric permanently vacuous. `ingestion/bigquery_client.py`'s
`fetch_patents_by_id()` now fetches the seed set's own examiner citations by ID after the
seed fetch, growing the corpus from ~290 to ~1,490 patents. This is the kind of scope change
this section anticipated ("changing this later... not a day-one decision") — driven by a
concrete requirement (the eval needing reachable ground truth), not scope creep.

## Changing this later

If `G06N3` turns out too small/large once real query volumes against BigQuery are known,
change `target_cpc_class` in `src/config/settings.py` — the ingestion query
in `src/ingestion/bigquery_client.py` reads it from settings, not a hardcoded
literal.
