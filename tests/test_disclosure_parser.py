import json
from unittest.mock import MagicMock

import pytest

from agents.disclosure_parser import parse_disclosure
from config.settings import Settings, get_settings


def _fake_response(content: str):
    response = MagicMock()
    response.choices[0].message.content = content
    return response


def _client_returning(*contents: str) -> MagicMock:
    client = MagicMock()
    client.chat_completion.side_effect = [_fake_response(c) for c in contents]
    return client


VALID_JSON = json.dumps(
    {
        "technical_field": "Neural network training regularization",
        "key_elements": ["dropout regularization", "backpropagation"],
        "candidate_cpc_classes": ["G06N3/08"],
    }
)


def test_parse_disclosure_returns_populated_disclosure():
    client = _client_returning(VALID_JSON)
    disclosure = parse_disclosure("A method that randomly disables neurons during training.", client=client)

    assert disclosure.raw_text == "A method that randomly disables neurons during training."
    assert disclosure.technical_field == "Neural network training regularization"
    assert disclosure.key_elements == ["dropout regularization", "backpropagation"]
    assert disclosure.candidate_cpc_classes == ["G06N3/08"]


def test_parse_disclosure_calls_groq_with_json_mode_and_correct_model():
    client = _client_returning(VALID_JSON)
    settings = Settings(groq_model="test-model")

    parse_disclosure("some disclosure", client=client, settings=settings)

    call_kwargs = client.chat_completion.call_args.kwargs
    assert call_kwargs["model"] == "test-model"
    assert call_kwargs["response_format"] == {"type": "json_object"}


def test_parse_disclosure_retries_on_malformed_json_then_succeeds():
    client = _client_returning("not json at all", VALID_JSON)
    disclosure = parse_disclosure("x", client=client)

    assert disclosure.technical_field == "Neural network training regularization"
    assert client.chat_completion.call_count == 2


def test_parse_disclosure_retries_on_missing_key_then_succeeds():
    missing_key_json = json.dumps({"technical_field": "x", "key_elements": []})  # no candidate_cpc_classes
    client = _client_returning(missing_key_json, VALID_JSON)

    disclosure = parse_disclosure("x", client=client)

    assert disclosure.candidate_cpc_classes == ["G06N3/08"]
    assert client.chat_completion.call_count == 2


def test_parse_disclosure_raises_after_exhausting_retries():
    client = _client_returning("not json", "still not json")

    with pytest.raises(ValueError, match="failed to produce valid output"):
        parse_disclosure("x", client=client)

    assert client.chat_completion.call_count == 2


@pytest.mark.integration
def test_parse_disclosure_live_groq_call():
    settings = get_settings()
    if not settings.groq_api_keys:
        pytest.skip("GROQ_API_KEY not configured — set up .env to run this against live Groq")

    disclosure = parse_disclosure(
        "A convolutional neural network for image classification that applies dropout "
        "regularization to fully-connected layers during training to reduce overfitting.",
        settings=settings,
    )

    assert disclosure.technical_field
    assert len(disclosure.key_elements) > 0
    assert len(disclosure.candidate_cpc_classes) > 0
    assert any("dropout" in e.lower() for e in disclosure.key_elements)
