"""Disclosure-parser agent (todo.md section 2): free-text invention disclosure → structured
`InventionDisclosure` (technical field, key elements, candidate CPC classes).

This is the first real LLM-backed pipeline stage. Its output feeds the search agent's query
construction — see `agents/search_agent.py` and `api/pipeline.py`.
"""

from __future__ import annotations

from agents.groq_client import RotatingGroqClient, build_groq_client
from agents.groq_json import request_json
from config.settings import Settings, get_settings
from schema import InventionDisclosure
from tracing import traced

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


@traced("disclosure_parser")
def parse_disclosure(
    raw_text: str,
    client: RotatingGroqClient | None = None,
    settings: Settings | None = None,
) -> InventionDisclosure:
    """Extract technical_field/key_elements/candidate_cpc_classes from free text via Groq."""
    settings = settings or get_settings()
    client = client or build_groq_client(settings)

    return request_json(
        client,
        settings,
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=f"Invention disclosure:\n\n{raw_text}",
        validate=lambda parsed: InventionDisclosure(
            raw_text=raw_text,
            technical_field=parsed["technical_field"],
            key_elements=parsed["key_elements"],
            candidate_cpc_classes=parsed["candidate_cpc_classes"],
        ),
    )
