"""Local fixture corpus loader — stands in for `bigquery_client.py` until a GCP project and
BigQuery access to Google Patents Public Data are available (see log.md, 2026-08-16 entry).

Returns the same `list[Patent]` shape the real BigQuery ingestion path will, so retrieval
code built against this fixture corpus doesn't need to change when the real client lands —
only the source of `raw_patents` changes, not what downstream code does with them.
"""

from __future__ import annotations

import json
from functools import lru_cache

from ingestion.chunking import split_claims
from schema import Citation, Patent

_FIXTURE_FILE = "sample_patents.json"


def _fixture_path() -> str:
    # tests/ isn't a package under src/, so resolve relative to the repo layout rather than
    # importlib.resources (which expects an installed package).
    from pathlib import Path

    return str(Path(__file__).resolve().parents[2] / "tests" / "fixtures" / _FIXTURE_FILE)


@lru_cache
def load_fixture_patents() -> list[Patent]:
    """Load and parse the hand-built fixture corpus into `Patent` objects.

    Cached since the fixture file is static within a process and re-parsing (including
    claim splitting) on every call would be wasted work for retrieval index builds.
    """
    with open(_fixture_path(), encoding="utf-8") as f:
        raw_patents = json.load(f)

    patents = []
    for raw in raw_patents:
        claims = split_claims(raw["claims_text"])
        citations = [Citation(**c) for c in raw.get("citations", [])]
        patents.append(
            Patent(
                patent_id=raw["patent_id"],
                title=raw["title"],
                abstract=raw["abstract"],
                claims=claims,
                cpc_codes=raw["cpc_codes"],
                assignees=raw["assignees"],
                publication_date=raw.get("publication_date"),
                citations=citations,
            )
        )
    return patents
