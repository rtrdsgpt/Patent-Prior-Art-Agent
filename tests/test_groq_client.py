from unittest.mock import MagicMock

import httpx
import pytest
from groq import RateLimitError

from agents.groq_client import NoAvailableGroqKeyError, RotatingGroqClient, build_groq_client
from config.settings import Settings


def _rate_limit_error() -> RateLimitError:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(status_code=429, request=request)
    return RateLimitError("rate limited", response=response, body=None)


def _make_client_stub(*, raises: bool = False, result="ok"):
    client = MagicMock()
    if raises:
        client.chat.completions.create.side_effect = _rate_limit_error()
    else:
        client.chat.completions.create.return_value = result
    return client


def test_requires_at_least_one_api_key():
    with pytest.raises(ValueError):
        RotatingGroqClient([])


def test_chat_completion_uses_first_key_when_not_rate_limited(monkeypatch):
    stub = _make_client_stub(result="response-a")
    monkeypatch.setattr("agents.groq_client.Groq", lambda api_key: stub)

    client = RotatingGroqClient(["key-a"])
    result = client.chat_completion(model="x", messages=[])

    assert result == "response-a"


def test_chat_completion_rotates_to_next_key_on_rate_limit(monkeypatch):
    stub_a = _make_client_stub(raises=True)
    stub_b = _make_client_stub(result="response-b")
    stubs = iter([stub_a, stub_b])
    monkeypatch.setattr("agents.groq_client.Groq", lambda api_key: next(stubs))

    client = RotatingGroqClient(["key-a", "key-b"])
    result = client.chat_completion(model="x", messages=[])

    assert result == "response-b"
    stub_a.chat.completions.create.assert_called_once()
    stub_b.chat.completions.create.assert_called_once()


def test_chat_completion_raises_when_all_keys_rate_limited(monkeypatch):
    monkeypatch.setattr("agents.groq_client.Groq", lambda api_key: _make_client_stub(raises=True))

    client = RotatingGroqClient(["key-a", "key-b", "key-c"])
    with pytest.raises(NoAvailableGroqKeyError):
        client.chat_completion(model="x", messages=[])


def test_rotation_is_sticky_across_calls(monkeypatch):
    # key-a rate-limited once; subsequent calls should start from key-b, not retry key-a.
    stub_a = _make_client_stub(raises=True)
    stub_b = _make_client_stub(result="response-b")
    stubs = iter([stub_a, stub_b])
    monkeypatch.setattr("agents.groq_client.Groq", lambda api_key: next(stubs))

    client = RotatingGroqClient(["key-a", "key-b"])
    client.chat_completion(model="x", messages=[])  # rotates past key-a
    stub_b.chat.completions.create.reset_mock()

    client.chat_completion(model="x", messages=[])  # should go straight to key-b

    stub_a.chat.completions.create.assert_called_once()  # still just the one call from before
    stub_b.chat.completions.create.assert_called_once()


def test_build_groq_client_reads_settings():
    settings = Settings(groq_api_key="key-a,key-b")
    client = build_groq_client(settings)
    assert len(client._clients) == 2


@pytest.mark.integration
def test_chat_completion_live_groq_call():
    from config.settings import get_settings

    settings = get_settings()
    if not settings.groq_api_keys:
        pytest.skip("GROQ_API_KEY not configured — set up .env to run this against live Groq")

    client = build_groq_client(settings)
    response = client.chat_completion(
        model=settings.groq_model,
        messages=[{"role": "user", "content": "Reply with exactly one word: OK"}],
        max_tokens=5,
    )

    assert "OK" in response.choices[0].message.content.upper()
