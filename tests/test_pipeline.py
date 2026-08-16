from unittest.mock import MagicMock

from patent_agent.api.pipeline import _build_chroma_client
from patent_agent.config.settings import Settings


def test_build_chroma_client_uses_http_client_when_host_set(monkeypatch):
    mock_http_client = MagicMock(name="HttpClient")
    monkeypatch.setattr("patent_agent.api.pipeline.chromadb.HttpClient", mock_http_client)

    settings = Settings(chroma_host="chroma", chroma_port=8123)
    _build_chroma_client(settings)

    mock_http_client.assert_called_once_with(host="chroma", port=8123)


def test_build_chroma_client_uses_persistent_client_when_host_unset(monkeypatch):
    mock_persistent_client = MagicMock(name="PersistentClient")
    monkeypatch.setattr("patent_agent.api.pipeline.chromadb.PersistentClient", mock_persistent_client)

    settings = Settings(chroma_host=None, chroma_persist_directory="some_dir")
    _build_chroma_client(settings)

    mock_persistent_client.assert_called_once_with(path="some_dir")
