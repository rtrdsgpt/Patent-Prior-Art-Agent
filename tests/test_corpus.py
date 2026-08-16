from config.settings import Settings
from ingestion.corpus import load_corpus, save_corpus
from ingestion.fixtures import load_fixture_patents


def test_load_corpus_falls_back_to_fixtures_when_no_cache(tmp_path):
    settings = Settings(corpus_cache_path=str(tmp_path / "does_not_exist.json"))
    patents = load_corpus(settings)
    assert patents == load_fixture_patents()


def test_save_then_load_corpus_roundtrips(tmp_path):
    settings = Settings(corpus_cache_path=str(tmp_path / "nested" / "corpus.json"))
    original = load_fixture_patents()[:3]

    cache_path = save_corpus(original, settings)
    assert cache_path.exists()

    loaded = load_corpus(settings)
    assert loaded == original


def test_save_corpus_creates_parent_directories(tmp_path):
    settings = Settings(corpus_cache_path=str(tmp_path / "a" / "b" / "c" / "corpus.json"))
    save_corpus([], settings)
    assert (tmp_path / "a" / "b" / "c" / "corpus.json").exists()


def test_load_corpus_empty_cache_returns_empty_list(tmp_path):
    settings = Settings(corpus_cache_path=str(tmp_path / "empty.json"))
    save_corpus([], settings)
    assert load_corpus(settings) == []
