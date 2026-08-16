"""Real ingestion from Google Patents Public Data (BigQuery) — see docs/cpc_scope.md and
log.md for why this data source over the live USPTO API, and why this module was paused
until a GCP project with billing + BigQuery API enabled and application-default credentials
were available.

Schema fields used here (`title_localized`, `abstract_localized`, `claims_localized`, `cpc`,
`assignee_harmonized`, `citation`) were confirmed against the live
`patents-public-data.patents.publications` table before writing this query, not assumed from
documentation — see the log.md entry for the exact `bq show --schema` check.
"""

from __future__ import annotations

from datetime import date

from google.cloud import bigquery

from patent_agent.config.settings import Settings, get_settings
from patent_agent.ingestion.chunking import split_claims
from patent_agent.schema import Citation, CitationCategory, Patent

_TABLE = "patents-public-data.patents.publications"

_QUERY = f"""
SELECT
  publication_number,
  (SELECT text FROM UNNEST(title_localized) WHERE language = 'en' LIMIT 1) AS title,
  (SELECT text FROM UNNEST(abstract_localized) WHERE language = 'en' LIMIT 1) AS abstract,
  (SELECT text FROM UNNEST(claims_localized) WHERE language = 'en' LIMIT 1) AS claims_text,
  ARRAY(SELECT code FROM UNNEST(cpc)) AS cpc_codes,
  ARRAY(SELECT name FROM UNNEST(assignee_harmonized)) AS assignees,
  publication_date,
  ARRAY(
    SELECT AS STRUCT publication_number AS cited_patent_id, category
    FROM UNNEST(citation)
    WHERE publication_number != ''
  ) AS citations
FROM `{_TABLE}`
WHERE country_code = 'US'
  AND EXISTS(SELECT 1 FROM UNNEST(cpc) AS c WHERE c.code LIKE @cpc_prefix)
  AND EXISTS(SELECT 1 FROM UNNEST(claims_localized) AS cl WHERE cl.language = 'en' AND cl.text != '')
LIMIT @corpus_size
"""

# citation.category's real values (CH2/SUP/ISR/SEA/APP/EXA/OPP/115/PRS/APL/FOP, per the
# field's BigQuery description) map onto only two of our three CitationCategory members —
# everything that isn't examiner- or applicant-cited prior art collapses to OTHER.
_CATEGORY_MAP = {
    "EXA": CitationCategory.EXAMINER,
    "APP": CitationCategory.APPLICANT,
}


def _parse_publication_date(raw: int | None) -> date | None:
    """`publication_date` is an INTEGER in YYYYMMDD form (e.g. 20210513), or 0/absent for
    unpublished/malformed records — those are treated as unknown rather than raising, since
    a missing publication date shouldn't fail ingestion of an otherwise-usable patent."""
    if not raw:
        return None
    try:
        return date(raw // 10000, (raw // 100) % 100, raw % 100)
    except ValueError:
        return None


def _row_to_patent(row: bigquery.table.Row) -> Patent | None:
    if not row["claims_text"]:
        return None

    claims = split_claims(row["claims_text"])
    if not claims:
        return None

    citations = [
        Citation(cited_patent_id=c["cited_patent_id"], category=_CATEGORY_MAP.get(c["category"], CitationCategory.OTHER))
        for c in row["citations"]
    ]

    return Patent(
        patent_id=row["publication_number"],
        title=row["title"] or "",
        abstract=row["abstract"] or "",
        claims=claims,
        cpc_codes=list(row["cpc_codes"]),
        assignees=list(row["assignees"]),
        publication_date=_parse_publication_date(row["publication_date"]),
        citations=citations,
    )


def fetch_patents(settings: Settings | None = None, client: bigquery.Client | None = None) -> list[Patent]:
    """Query Google Patents Public Data for up to `settings.corpus_size` US patents in
    `settings.target_cpc_class`, and parse them into `Patent` objects.

    Rows with no English claims text are skipped (`_row_to_patent` returns `None`) — a
    patent this pipeline can't chunk into claims is one the comparison agent can't use
    downstream, so there's no point carrying it through ingestion.
    """
    settings = settings or get_settings()
    client = client or bigquery.Client(project=settings.gcp_project_id)

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("cpc_prefix", "STRING", f"{settings.target_cpc_class}%"),
            bigquery.ScalarQueryParameter("corpus_size", "INT64", settings.corpus_size),
        ]
    )
    rows = client.query(_QUERY, job_config=job_config).result()

    patents = [_row_to_patent(row) for row in rows]
    return [p for p in patents if p is not None]
