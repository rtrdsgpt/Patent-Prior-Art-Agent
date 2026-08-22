from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from agents.groq_json import request_structured
from config.settings import Settings


class _Schema(BaseModel):
    x: int


def _client_returning(*results) -> MagicMock:
    """`results` entries that are `Exception` instances are raised instead of returned,
    so a test can interleave failures and successes across attempts."""
    client = MagicMock()
    side_effects = [r if isinstance(r, Exception) else r for r in results]
    client.with_structured_output.return_value.invoke.side_effect = side_effects
    return client


def test_request_structured_returns_validated_result_on_first_try():
    client = _client_returning(_Schema(x=1))
    result = request_structured(client, Settings(), "system", "user", schema=_Schema, validate=lambda parsed: parsed.x)
    assert result == 1


def test_request_structured_passes_schema_and_messages_to_client():
    client = _client_returning(_Schema(x=1))
    request_structured(client, Settings(), "system", "user", schema=_Schema, validate=lambda parsed: parsed.x)

    client.with_structured_output.assert_called_once_with(_Schema)
    messages = client.with_structured_output.return_value.invoke.call_args.args[0]
    assert messages[0] == {"role": "system", "content": "system"}
    assert messages[1] == {"role": "user", "content": "user"}


def test_request_structured_retries_when_invoke_raises():
    client = _client_returning(ValueError("bad tool call"), _Schema(x=2))
    result = request_structured(client, Settings(), "system", "user", schema=_Schema, validate=lambda parsed: parsed.x)
    assert result == 2
    assert client.with_structured_output.return_value.invoke.call_count == 2


def test_request_structured_retries_when_validate_raises():
    def validate(parsed: _Schema) -> int:
        if parsed.x != 3:
            raise ValueError("not the expected value")
        return parsed.x

    client = _client_returning(_Schema(x=1), _Schema(x=3))
    result = request_structured(client, Settings(), "system", "user", schema=_Schema, validate=validate)
    assert result == 3
    assert client.with_structured_output.return_value.invoke.call_count == 2


def test_request_structured_raises_after_exhausting_max_attempts():
    client = _client_returning(ValueError("bad"), ValueError("still bad"), ValueError("also bad"))
    with pytest.raises(ValueError, match="failed to produce valid output"):
        request_structured(client, Settings(), "system", "user", schema=_Schema, validate=lambda parsed: parsed.x, max_attempts=3)
    assert client.with_structured_output.return_value.invoke.call_count == 3


def test_request_structured_feeds_correction_message_back_to_model():
    client = _client_returning(ValueError("bad tool call"), _Schema(x=1))
    request_structured(client, Settings(), "system", "user", schema=_Schema, validate=lambda parsed: parsed.x)

    second_call_messages = client.with_structured_output.return_value.invoke.call_args_list[1].args[0]
    assert "wasn't valid" in second_call_messages[-1]["content"]
