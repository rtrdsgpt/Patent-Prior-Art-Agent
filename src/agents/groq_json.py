"""Shared "ask Groq for structured output, validate it, bounded-retry on failure" loop.

Extracted after `disclosure_parser.py` wrote this pattern once and `claims_parser.py`/
`comparison_agent.py` were both about to repeat it — the third agent about to duplicate a
loop is the right time to pull it out, not the first.

Uses `RotatingChatGroq.with_structured_output(schema)` (LangChain's structured-output
mechanism — tool-calling under the hood) rather than raw JSON-mode + manual `json.loads`.
This still isn't a *guarantee* the result satisfies whatever extra business-logic checks a
caller's own `validate` needs beyond the Pydantic schema itself (e.g. claims_parser.py checks
that every expected claim number got a response, which no schema alone can express) — so the
bounded-retry loop is still real, just operating on an already-parsed object instead of a
raw JSON string.
"""

from __future__ import annotations

import logging
from typing import Callable, TypeVar

from pydantic import BaseModel

from agents.groq_client import RotatingChatGroq
from config.settings import Settings

logger = logging.getLogger(__name__)

T = TypeVar("T")

_DEFAULT_MAX_ATTEMPTS = 2


def request_structured(
    client: RotatingChatGroq,
    settings: Settings,
    system_prompt: str,
    user_prompt: str,
    schema: type[BaseModel],
    validate: Callable[[BaseModel], T],
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
) -> T:
    """Call Groq via `with_structured_output(schema)` and run the result through `validate`,
    retrying (with a correction message appended) if it fails.

    Schema validation happens inside LangChain's structured-output mechanism itself — what
    `validate` catches here is everything beyond "matches the Pydantic shape": custom
    business-logic checks a schema alone can't express (e.g. "every expected claim number
    got a response"). Any exception `validate` raises is treated as "ask the model to try
    again," not just a specific error type.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    structured_client = client.with_structured_output(schema)

    last_error: Exception | None = None
    for attempt in range(max_attempts):
        try:
            parsed = structured_client.invoke(messages)
            return validate(parsed)
        except Exception as exc:  # noqa: BLE001 - any failure here means "retry with the model," see docstring
            logger.warning("Structured request attempt %d/%d produced invalid output: %s", attempt + 1, max_attempts, exc)
            last_error = exc
            messages.append({"role": "user", "content": f"That wasn't valid: {exc}. Try again, adjusting your response accordingly."})

    raise ValueError(f"Groq structured request failed to produce valid output after {max_attempts} attempts") from last_error
