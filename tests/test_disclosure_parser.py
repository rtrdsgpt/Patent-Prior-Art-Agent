from unittest.mock import MagicMock

import pytest

from agents.disclosure_parser import _DisclosureExtraction, parse_disclosure
from config.settings import get_settings

VALID_EXTRACTION = _DisclosureExtraction(
    technical_field="Neural network training regularization",
    key_elements=["dropout regularization", "backpropagation"],
    candidate_cpc_classes=["G06N3/08"],
)


def _client_returning(*results) -> MagicMock:
    client = MagicMock()
    client.with_structured_output.return_value.invoke.side_effect = list(results)
    return client


def test_parse_disclosure_returns_populated_disclosure():
    client = _client_returning(VALID_EXTRACTION)
    disclosure = parse_disclosure("A method that randomly disables neurons during training.", client=client)

    assert disclosure.raw_text == "A method that randomly disables neurons during training."
    assert disclosure.technical_field == "Neural network training regularization"
    assert disclosure.key_elements == ["dropout regularization", "backpropagation"]
    assert disclosure.candidate_cpc_classes == ["G06N3/08"]


def test_parse_disclosure_uses_correct_schema():
    client = _client_returning(VALID_EXTRACTION)
    parse_disclosure("some disclosure", client=client)

    client.with_structured_output.assert_called_once_with(_DisclosureExtraction)


def test_parse_disclosure_handles_missing_optional_fields_via_defaults():
    minimal = _DisclosureExtraction(technical_field="x")  # key_elements/candidate_cpc_classes default to []
    client = _client_returning(minimal)

    disclosure = parse_disclosure("x", client=client)

    assert disclosure.key_elements == []
    assert disclosure.candidate_cpc_classes == []


def test_parse_disclosure_retries_when_invoke_raises_then_succeeds():
    client = _client_returning(ValueError("model produced an unparseable tool call"), VALID_EXTRACTION)
    disclosure = parse_disclosure("x", client=client)

    assert disclosure.technical_field == "Neural network training regularization"
    assert client.with_structured_output.return_value.invoke.call_count == 2


def test_parse_disclosure_raises_after_exhausting_retries():
    client = _client_returning(ValueError("bad"), ValueError("still bad"))

    with pytest.raises(ValueError, match="failed to produce valid output"):
        parse_disclosure("x", client=client)

    assert client.with_structured_output.return_value.invoke.call_count == 2


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
