import pytest
from google.cloud.bigquery.table import Row

from patent_agent.config.settings import Settings, get_settings
from patent_agent.ingestion.bigquery_client import _parse_publication_date, _row_to_patent, fetch_patents


def _make_row(**fields) -> Row:
    values = tuple(fields.values())
    field_to_index = {name: i for i, name in enumerate(fields)}
    return Row(values, field_to_index)


def _base_row_fields(**overrides) -> dict:
    fields = dict(
        publication_number="US-1234567-A1",
        title="A method for widgets",
        abstract="An abstract about widgets.",
        claims_text="1. A method comprising doing a thing.\n\n2. The method of claim 1, further comprising doing another thing.\n",
        cpc_codes=["G06N3/08"],
        assignees=["Widget Corp"],
        publication_date=20210513,
        citations=[],
    )
    fields.update(overrides)
    return fields


def test_parse_publication_date_valid():
    assert _parse_publication_date(20210513).isoformat() == "2021-05-13"


def test_parse_publication_date_none_for_missing():
    assert _parse_publication_date(None) is None
    assert _parse_publication_date(0) is None


def test_parse_publication_date_none_for_malformed():
    assert _parse_publication_date(20219999) is None


def test_row_to_patent_parses_claims_and_metadata():
    patent = _row_to_patent(_make_row(**_base_row_fields()))
    assert patent.patent_id == "US-1234567-A1"
    assert patent.title == "A method for widgets"
    assert len(patent.claims) == 2
    assert patent.cpc_codes == ["G06N3/08"]
    assert patent.publication_date.isoformat() == "2021-05-13"


def test_row_to_patent_returns_none_when_no_claims_text():
    patent = _row_to_patent(_make_row(**_base_row_fields(claims_text="")))
    assert patent is None


def test_row_to_patent_returns_none_when_claims_text_does_not_parse_into_claims():
    # No sequential "N. " boundary at all, so split_claims yields nothing.
    patent = _row_to_patent(_make_row(**_base_row_fields(claims_text="not a claims section")))
    assert patent is None


def test_row_to_patent_maps_examiner_and_applicant_citation_categories():
    row = _make_row(
        **_base_row_fields(
            citations=[
                {"cited_patent_id": "US-1111111-A1", "category": "EXA"},
                {"cited_patent_id": "US-2222222-A1", "category": "APP"},
            ]
        )
    )
    patent = _row_to_patent(row)
    assert patent.examiner_cited_patent_ids == {"US-1111111-A1"}


def test_row_to_patent_maps_unrecognized_citation_category_to_other():
    from patent_agent.schema import CitationCategory

    row = _make_row(**_base_row_fields(citations=[{"cited_patent_id": "US-3333333-A1", "category": "OPP"}]))
    patent = _row_to_patent(row)
    assert patent.citations[0].category == CitationCategory.OTHER


def test_row_to_patent_handles_missing_title_and_abstract():
    patent = _row_to_patent(_make_row(**_base_row_fields(title=None, abstract=None)))
    assert patent.title == ""
    assert patent.abstract == ""


@pytest.mark.integration
def test_fetch_patents_live_bigquery_query():
    settings = get_settings()
    if not settings.gcp_project_id:
        pytest.skip("GCP_PROJECT_ID not configured — set up .env to run this against live BigQuery")

    patents = fetch_patents(Settings(gcp_project_id=settings.gcp_project_id, target_cpc_class="G06N3", corpus_size=3))

    assert 1 <= len(patents) <= 3
    for patent in patents:
        assert patent.patent_id.startswith("US")
        assert any(code.startswith("G06N3") for code in patent.cpc_codes)
        assert len(patent.claims) > 0
