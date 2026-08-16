from ingestion.fixtures import load_fixture_patents
from schema import CitationCategory


def test_load_fixture_patents_returns_patents():
    patents = load_fixture_patents()
    assert len(patents) == 8


def test_load_fixture_patents_parses_claims_for_every_patent():
    patents = load_fixture_patents()
    assert all(len(p.claims) >= 4 for p in patents)


def test_load_fixture_patents_identifies_independent_claims():
    patents = load_fixture_patents()
    dropout_patent = next(p for p in patents if p.patent_id == "US10000001B2")
    assert dropout_patent.independent_claims[0].claim_number == 1
    assert dropout_patent.independent_claims[0].is_independent is True


def test_load_fixture_patents_parses_examiner_citations():
    patents = load_fixture_patents()
    cnn_patent = next(p for p in patents if p.patent_id == "US10000002B2")
    assert cnn_patent.examiner_cited_patent_ids == {"US10000001B2"}


def test_load_fixture_patents_applicant_citations_excluded_from_examiner_set():
    patents = load_fixture_patents()
    gan_patent = next(p for p in patents if p.patent_id == "US10000006B2")
    citation_categories = {c.category for c in gan_patent.citations}
    assert CitationCategory.APPLICANT in citation_categories
    assert "US10000002B2" not in gan_patent.examiner_cited_patent_ids


def test_load_fixture_patents_is_cached():
    assert load_fixture_patents() is load_fixture_patents()
