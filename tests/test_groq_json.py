from unittest.mock import MagicMock

import pytest

from agents.groq_json import request_json
from config.settings import Settings


def _fake_response(content: str):
    response = MagicMock()
    response.choices[0].message.content = content
    return response


def _client_returning(*contents: str) -> MagicMock:
    client = MagicMock()
    client.chat_completion.side_effect = [_fake_response(c) for c in contents]
    return client


def test_request_json_returns_validated_result_on_first_try():
    client = _client_returning('{"x": 1}')
    result = request_json(client, Settings(), "system", "user", validate=lambda d: d["x"])
    assert result == 1


def test_request_json_passes_json_mode_and_model_to_client():
    client = _client_returning('{"x": 1}')
    settings = Settings(groq_model="my-model")
    request_json(client, settings, "system", "user", validate=lambda d: d["x"])

    kwargs = client.chat_completion.call_args.kwargs
    assert kwargs["model"] == "my-model"
    assert kwargs["response_format"] == {"type": "json_object"}


def test_request_json_retries_on_invalid_json_syntax():
    client = _client_returning("not json", '{"x": 2}')
    result = request_json(client, Settings(), "system", "user", validate=lambda d: d["x"])
    assert result == 2
    assert client.chat_completion.call_count == 2


def test_request_json_retries_when_validate_raises():
    def validate(parsed: dict) -> int:
        if "x" not in parsed:
            raise KeyError("x")
        return parsed["x"]

    client = _client_returning('{"y": 1}', '{"x": 3}')
    result = request_json(client, Settings(), "system", "user", validate=validate)
    assert result == 3
    assert client.chat_completion.call_count == 2


def test_request_json_raises_after_exhausting_max_attempts():
    client = _client_returning("bad", "still bad", "also bad")
    with pytest.raises(ValueError, match="failed to produce valid output"):
        request_json(client, Settings(), "system", "user", validate=lambda d: d["x"], max_attempts=3)
    assert client.chat_completion.call_count == 3


def test_request_json_feeds_correction_message_back_to_model():
    client = _client_returning("not json", '{"x": 1}')
    request_json(client, Settings(), "system", "user", validate=lambda d: d["x"])

    second_call_messages = client.chat_completion.call_args_list[1].kwargs["messages"]
    assert second_call_messages[-2]["content"] == "not json"
    assert "wasn't valid" in second_call_messages[-1]["content"]
