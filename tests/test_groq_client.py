from unittest.mock import MagicMock

import httpx
import pytest
from groq import RateLimitError
from pydantic import BaseModel

from agents.groq_client import NoAvailableGroqKeyError, RotatingChatGroq, build_groq_client
from config.settings import Settings


def _rate_limit_error() -> RateLimitError:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(status_code=429, request=request)
    return RateLimitError("rate limited", response=response, body=None)


def _make_model_stub(*, raises: bool = False, result="ok", structured_result="structured-ok"):
    model = MagicMock()
    if raises:
        model.invoke.side_effect = _rate_limit_error()
        model.with_structured_output.return_value.invoke.side_effect = _rate_limit_error()
    else:
        model.invoke.return_value = result
        model.with_structured_output.return_value.invoke.return_value = structured_result
    return model


def test_requires_at_least_one_api_key():
    with pytest.raises(ValueError):
        RotatingChatGroq([], model="x")


def test_invoke_uses_first_key_when_not_rate_limited(monkeypatch):
    stub = _make_model_stub(result="response-a")
    monkeypatch.setattr("agents.groq_client.ChatGroq", lambda model, api_key, temperature: stub)

    client = RotatingChatGroq(["key-a"], model="x")
    result = client.invoke(messages=[])

    assert result == "response-a"


def test_invoke_rotates_to_next_key_on_rate_limit(monkeypatch):
    stub_a = _make_model_stub(raises=True)
    stub_b = _make_model_stub(result="response-b")
    stubs = iter([stub_a, stub_b])
    monkeypatch.setattr("agents.groq_client.ChatGroq", lambda model, api_key, temperature: next(stubs))

    client = RotatingChatGroq(["key-a", "key-b"], model="x")
    result = client.invoke(messages=[])

    assert result == "response-b"
    stub_a.invoke.assert_called_once()
    stub_b.invoke.assert_called_once()


def test_invoke_raises_when_all_keys_rate_limited(monkeypatch):
    monkeypatch.setattr("agents.groq_client.ChatGroq", lambda model, api_key, temperature: _make_model_stub(raises=True))

    client = RotatingChatGroq(["key-a", "key-b", "key-c"], model="x")
    with pytest.raises(NoAvailableGroqKeyError):
        client.invoke(messages=[])


def test_rotation_is_sticky_across_calls(monkeypatch):
    # key-a rate-limited once; subsequent calls should start from key-b, not retry key-a.
    stub_a = _make_model_stub(raises=True)
    stub_b = _make_model_stub(result="response-b")
    stubs = iter([stub_a, stub_b])
    monkeypatch.setattr("agents.groq_client.ChatGroq", lambda model, api_key, temperature: next(stubs))

    client = RotatingChatGroq(["key-a", "key-b"], model="x")
    client.invoke(messages=[])  # rotates past key-a
    stub_b.invoke.reset_mock()

    client.invoke(messages=[])  # should go straight to key-b

    stub_a.invoke.assert_called_once()  # still just the one call from before
    stub_b.invoke.assert_called_once()


def test_with_structured_output_invoke_rotates_on_rate_limit(monkeypatch):
    stub_a = _make_model_stub(raises=True)
    stub_b = _make_model_stub(structured_result="parsed-b")
    stubs = iter([stub_a, stub_b])
    monkeypatch.setattr("agents.groq_client.ChatGroq", lambda model, api_key, temperature: next(stubs))

    class Schema(BaseModel):
        x: int

    client = RotatingChatGroq(["key-a", "key-b"], model="x")
    result = client.with_structured_output(Schema).invoke(messages=[])

    assert result == "parsed-b"
    stub_a.with_structured_output.return_value.invoke.assert_called_once()
    stub_b.with_structured_output.return_value.invoke.assert_called_once()


def test_build_groq_client_reads_settings():
    settings = Settings(groq_api_key="key-a,key-b")
    client = build_groq_client(settings)
    assert len(client._models) == 2


@pytest.mark.integration
def test_invoke_live_groq_call():
    from config.settings import get_settings

    settings = get_settings()
    if not settings.groq_api_keys:
        pytest.skip("GROQ_API_KEY not configured — set up .env to run this against live Groq")

    client = build_groq_client(settings)
    response = client.invoke(messages=[{"role": "user", "content": "Reply with exactly one word: OK"}])

    assert "OK" in response.content.upper()


@pytest.mark.integration
def test_with_structured_output_live_groq_call():
    from config.settings import get_settings

    settings = get_settings()
    if not settings.groq_api_keys:
        pytest.skip("GROQ_API_KEY not configured — set up .env to run this against live Groq")

    class Extraction(BaseModel):
        answer: int

    client = build_groq_client(settings)
    result = client.with_structured_output(Extraction).invoke(messages=[{"role": "user", "content": "What is 2+2? Respond with the schema."}])

    assert result.answer == 4
