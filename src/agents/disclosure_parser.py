"""Disclosure-parser agent (todo.md section 2): free-text invention disclosure → structured
`InventionDisclosure` (technical field, key elements, candidate CPC classes).

This is the first real LLM-backed pipeline stage. Its output feeds the search agent's query
construction (replacing `api/pipeline.py`'s current placeholder of using the raw disclosure
text directly as the retrieval query — see that module's docstring).
"""

from __future__ import annotations

import json
import logging

from pydantic import ValidationError

from agents.groq_client import RotatingGroqClient, build_groq_client
from config.settings import Settings, get_settings
from schema import InventionDisclosure

logger = logging.getLogger(__name__)

_MAX_ATTEMPTS = 2  # one retry if the model's JSON doesn't validate against our schema

_SYSTEM_PROMPT = """You are a patent analyst extracting structured information from a free-text \
invention disclosure. Respond with ONLY a JSON object (no prose, no markdown fences) with exactly \
these three keys:

- "technical_field": a short (one sentence) description of the invention's technical field.
- "key_elements": an array of strings, each naming one distinct technical component, step, or \
feature the invention actually includes. Extract only what's described, don't invent elements. \
Prefer specific, concrete phrases (e.g. "dropout regularization during training") over vague ones \
(e.g. "a training technique").
- "candidate_cpc_classes": an array of 1-3 plausible CPC (Cooperative Patent Classification) \
class codes (e.g. "G06N3/08") this invention would likely be classified under, your best-effort \
guess based on the technical field."""


def _build_user_prompt(raw_text: str) -> str:
    return f"Invention disclosure:\n\n{raw_text}"


def parse_disclosure(
    raw_text: str,
    client: RotatingGroqClient | None = None,
    settings: Settings | None = None,
) -> InventionDisclosure:
    """Extract technical_field/key_elements/candidate_cpc_classes from free text via Groq.

    Bounded retry (`_MAX_ATTEMPTS`): JSON-mode guarantees syntactically valid JSON, not that
    it matches our schema (e.g. the model could return a string where we expect a list) — a
    single retry with the validation error appended to the prompt handles the occasional
    malformed response without looping indefinitely on a model that's consistently wrong.
    """
    settings = settings or get_settings()
    client = client or build_groq_client(settings)

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_prompt(raw_text)},
    ]

    last_error: Exception | None = None
    for attempt in range(_MAX_ATTEMPTS):
        response = client.chat_completion(
            model=settings.groq_model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0,
        )
        content = response.choices[0].message.content

        try:
            parsed = json.loads(content)
            return InventionDisclosure(
                raw_text=raw_text,
                technical_field=parsed["technical_field"],
                key_elements=parsed["key_elements"],
                candidate_cpc_classes=parsed["candidate_cpc_classes"],
            )
        except (json.JSONDecodeError, ValidationError, KeyError, TypeError) as exc:
            logger.warning("Disclosure-parser attempt %d/%d produced invalid output: %s", attempt + 1, _MAX_ATTEMPTS, exc)
            last_error = exc
            messages.append({"role": "assistant", "content": content})
            messages.append(
                {
                    "role": "user",
                    "content": f"That wasn't valid: {exc}. Respond again with ONLY the corrected JSON object.",
                }
            )

    raise ValueError(f"Disclosure-parser failed to produce valid output after {_MAX_ATTEMPTS} attempts") from last_error
