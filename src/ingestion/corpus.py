"""The corpus the running pipeline actually indexes: a real ingested BigQuery corpus if one
has been cached to disk, falling back to the hand-built fixture set otherwise.

Caching to disk (rather than querying BigQuery on every process start) matters for two
reasons: cost/latency (re-querying ~300 patents' worth of claims text on every restart is
wasteful), and reproducibility (retrieval results should be stable across restarts, not
silently shift if BigQuery's underlying data changes between runs). Run
`python -m ingestion.ingest_corpus` to (re)generate the cache; nothing here
calls BigQuery directly.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from config.settings import Settings, get_settings
from ingestion.fixtures import load_fixture_patents
from schema import Patent

logger = logging.getLogger(__name__)


def load_corpus(settings: Settings | None = None) -> list[Patent]:
    """Load the ingested corpus cache if it exists, else fall back to the fixture corpus.

    The fallback is deliberate, not a silent failure: this project needs to keep working
    (retrieval, the API, MCP tools, tests) for anyone without BigQuery credentials or who
    hasn't run the ingestion script yet, per the same reasoning as `ingestion/fixtures.py`.
    """
    settings = settings or get_settings()
    cache_path = Path(settings.corpus_cache_path)

    if not cache_path.exists():
        logger.warning(
            "No ingested corpus cache at %s — falling back to the %d-patent fixture corpus. "
            "Run `python -m ingestion.ingest_corpus` to ingest a real corpus.",
            cache_path,
            len(load_fixture_patents()),
        )
        return load_fixture_patents()

    with cache_path.open(encoding="utf-8") as f:
        raw_patents = json.load(f)
    return [Patent.model_validate(raw) for raw in raw_patents]


def save_corpus(patents: list[Patent], settings: Settings | None = None) -> Path:
    """Write `patents` to the corpus cache path, creating parent directories as needed."""
    settings = settings or get_settings()
    cache_path = Path(settings.corpus_cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    with cache_path.open("w", encoding="utf-8") as f:
        json.dump([p.model_dump(mode="json") for p in patents], f, indent=2)

    return cache_path
