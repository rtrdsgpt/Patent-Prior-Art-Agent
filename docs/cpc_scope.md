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
- **Real ground truth exists.** Google Patents Public Data's `citation.category = "EXA"`
  field gives examiner-cited prior art for patents in this class, which is what makes the
  recall@k evaluation (section 5 of `todo.md`) possible at all — this was checked before
  picking the class, not assumed.

## Scope discipline

Per the portfolio plan, don't expand CPC coverage until the first working slice — one
class, a small indexed corpus, search → compare → verify working end-to-end — is real.
Widening to sibling classes (e.g. `G06N20` — machine learning generally) is a natural
second-slice expansion once the pipeline works, not a day-one decision.

## Changing this later

If `G06N3` turns out too small/large once real query volumes against BigQuery are known,
change `target_cpc_class` in `src/patent_agent/config/settings.py` — the ingestion query
in `src/patent_agent/ingestion/bigquery_client.py` reads it from settings, not a hardcoded
literal.
