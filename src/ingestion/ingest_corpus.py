"""CLI entry point: pull `settings.corpus_size` patents from BigQuery, then (unless
`settings.expand_corpus_with_citations` is disabled) expand the corpus by fetching the seed
set's own examiner-cited patents by ID, and cache the combined result to
`settings.corpus_cache_path` for `ingestion/corpus.py` to load.

    python -m ingestion.ingest_corpus

This is a one-shot manual script for now, not the scheduled Airflow DAG described in
todo.md section 4 — see log.md for why that's deferred rather than built on top of a
single manual run.
"""

from __future__ import annotations

import logging

from config.settings import get_settings
from ingestion.bigquery_client import fetch_patents, fetch_patents_by_id
from ingestion.corpus import save_corpus

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    settings = get_settings()
    logger.info("Fetching up to %d patents in CPC class %s from BigQuery...", settings.corpus_size, settings.target_cpc_class)

    patents = fetch_patents(settings)
    logger.info("Fetched %d seed patents with parseable claims.", len(patents))

    if settings.expand_corpus_with_citations:
        corpus_ids = {p.patent_id for p in patents}
        cited_ids = sorted({cid for p in patents for cid in p.examiner_cited_patent_ids} - corpus_ids)
        logger.info("Fetching %d examiner-cited patents not already in the seed corpus...", len(cited_ids))

        cited_patents = fetch_patents_by_id(cited_ids, settings)
        logger.info("Fetched %d of those %d citations (rest had no usable English claims text).", len(cited_patents), len(cited_ids))
        patents = patents + cited_patents

    cache_path = save_corpus(patents, settings)
    logger.info("Wrote %d total patents to corpus cache at %s", len(patents), cache_path)


if __name__ == "__main__":
    main()
