import json
from unittest.mock import MagicMock

import pytest

from agents.claims_parser import parse_claim_elements
from config.settings import get_settings
from ingestion.fixtures import load_fixture_patents
from schema import Claim, Patent


def _fake_response(content: str):
    response = MagicMock()
    response.choices[0].message.content = content
    return response


def _client_returning(*contents: str) -> MagicMock:
    client = MagicMock()
    client.chat_completion.side_effect = [_fake_response(c) for c in contents]
    return client


def _patent(claims: list[Claim]) -> Patent:
    return Patent(
        patent_id="US1234567A1",
        title="Test patent",
        abstract="An abstract.",
        claims=claims,
        cpc_codes=["G06N3/08"],
        assignees=["Test Corp"],
    )


TWO_INDEPENDENT_CLAIMS = [
    Claim(claim_number=1, text="A method comprising: doing X; and doing Y.", is_independent=True),
    Claim(claim_number=2, text="The method of claim 1, further comprising doing Z.", is_independent=False, depends_on=1),
    Claim(claim_number=3, text="A system comprising: a memory; and a processor.", is_independent=True),
]

VALID_RESPONSE = json.dumps(
    {
        "claims": [
            {"claim_number": 1, "elements": ["doing X", "doing Y"]},
            {"claim_number": 3, "elements": ["a memory", "a processor"]},
        ]
    }
)


def test_parse_claim_elements_returns_empty_dict_for_no_independent_claims():
    patent = _patent([Claim(claim_number=1, text="dependent-only", is_independent=False, depends_on=None)])
    client = MagicMock()

    result = parse_claim_elements(patent, client=client)

    assert result == {}
    client.chat_completion.assert_not_called()


def test_parse_claim_elements_maps_claim_number_to_elements():
    patent = _patent(TWO_INDEPENDENT_CLAIMS)
    client = _client_returning(VALID_RESPONSE)

    result = parse_claim_elements(patent, client=client)

    assert result == {1: ["doing X", "doing Y"], 3: ["a memory", "a processor"]}


def test_parse_claim_elements_batches_all_independent_claims_into_one_call():
    patent = _patent(TWO_INDEPENDENT_CLAIMS)
    client = _client_returning(VALID_RESPONSE)

    parse_claim_elements(patent, client=client)

    assert client.chat_completion.call_count == 1
    user_prompt = client.chat_completion.call_args.kwargs["messages"][1]["content"]
    assert "Claim 1" in user_prompt
    assert "Claim 3" in user_prompt
    assert "Claim 2" not in user_prompt  # dependent claim excluded


def test_parse_claim_elements_retries_when_a_claim_is_missing_from_response():
    incomplete = json.dumps({"claims": [{"claim_number": 1, "elements": ["doing X"]}]})  # missing claim 3
    client = _client_returning(incomplete, VALID_RESPONSE)

    result = parse_claim_elements(_patent(TWO_INDEPENDENT_CLAIMS), client=client)

    assert result == {1: ["doing X", "doing Y"], 3: ["a memory", "a processor"]}
    assert client.chat_completion.call_count == 2


@pytest.mark.integration
def test_parse_claim_elements_live_groq_call():
    settings = get_settings()
    if not settings.groq_api_keys:
        pytest.skip("GROQ_API_KEY not configured — set up .env to run this against live Groq")

    patent = next(p for p in load_fixture_patents() if p.patent_id == "US10000001B2")  # dropout patent
    result = parse_claim_elements(patent, settings=settings)

    assert set(result.keys()) == {c.claim_number for c in patent.independent_claims}
    assert any("dropout" in e.lower() or "deactivat" in e.lower() for elements in result.values() for e in elements)
