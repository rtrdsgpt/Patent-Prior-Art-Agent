"""CLI entry point: pull `settings.corpus_size` patents from BigQuery and cache them to
`settings.corpus_cache_path` for `ingestion/corpus.py` to load.

    python -m ingestion.ingest_corpus

This is a one-shot manual script for now, not the scheduled Airflow DAG described in
todo.md section 4 — see log.md for why that's deferred rather than built on top of a
single manual run.
"""

from __future__ import annotations

import logging

from config.settings import get_settings
from ingestion.bigquery_client import fetch_patents
from ingestion.corpus import save_corpus

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    settings = get_settings()
    logger.info("Fetching up to %d patents in CPC class %s from BigQuery...", settings.corpus_size, settings.target_cpc_class)

    patents = fetch_patents(settings)
    logger.info("Fetched %d patents with parseable claims.", len(patents))

    cache_path = save_corpus(patents, settings)
    logger.info("Wrote corpus cache to %s", cache_path)


if __name__ == "__main__":
    main()
