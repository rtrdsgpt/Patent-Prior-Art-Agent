"""Shared "ask Groq for JSON, validate it, bounded-retry on failure" loop.

Extracted after `disclosure_parser.py` wrote this pattern once and `claims_parser.py`/
`comparison_agent.py`/`risk_report_agent.py` were all about to repeat it — the third
agent about to duplicate a loop is the right time to pull it out, not the first.
"""

from __future__ import annotations

import json
import logging
from typing import Callable, TypeVar

from agents.groq_client import RotatingGroqClient
from config.settings import Settings

logger = logging.getLogger(__name__)

T = TypeVar("T")

_DEFAULT_MAX_ATTEMPTS = 2


def request_json(
    client: RotatingGroqClient,
    settings: Settings,
    system_prompt: str,
    user_prompt: str,
    validate: Callable[[dict], T],
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
) -> T:
    """Call Groq in JSON mode and run the parsed JSON through `validate`, retrying with the
    validation error fed back to the model if it fails.

    JSON mode only guarantees syntactically valid JSON, not that it matches whatever shape
    `validate` expects (e.g. a field could be a string where a list is expected, or a key
    could be missing) — `validate` should raise on anything not usable, and any exception it
    raises is treated as "ask the model to try again," not just `ValueError`.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    last_error: Exception | None = None
    for attempt in range(max_attempts):
        response = client.chat_completion(
            model=settings.groq_model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0,
        )
        content = response.choices[0].message.content

        try:
            parsed = json.loads(content)
            return validate(parsed)
        except Exception as exc:  # noqa: BLE001 - any failure here means "retry with the model," see docstring
            logger.warning("JSON request attempt %d/%d produced invalid output: %s", attempt + 1, max_attempts, exc)
            last_error = exc
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": f"That wasn't valid: {exc}. Respond again with ONLY the corrected JSON object."})

    raise ValueError(f"Groq JSON request failed to produce valid output after {max_attempts} attempts") from last_error
