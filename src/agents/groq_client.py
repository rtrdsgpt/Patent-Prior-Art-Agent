"""Rotating Groq client: cycles across multiple API keys on a rate limit.

Groq's free tier has fairly tight per-key rate limits, and the agent pipeline (todo.md
section 2) will make several LLM calls per job across multiple agent stages — a single
free-tier key would throttle that quickly. The keys provided for this project are multiple
free-tier keys specifically for that reason, so this rotates to the next one whenever the
current one is rate-limited, rather than using only the first and leaving the rest idle.

Bounded to at most one attempt per configured key, same bounded-retry discipline as todo.md's
note to "reuse the bounded-retry pattern from Exporter Crawl's discovery.py" for the search
agent — retry with a clear stopping point, never loop indefinitely.
"""

from __future__ import annotations

import logging

from groq import Groq, RateLimitError

from config.settings import Settings, get_settings

logger = logging.getLogger(__name__)


class NoAvailableGroqKeyError(RuntimeError):
    """Raised when every configured Groq API key is rate-limited."""


class RotatingGroqClient:
    def __init__(self, api_keys: list[str]) -> None:
        if not api_keys:
            raise ValueError("RotatingGroqClient requires at least one API key")
        self._clients = [Groq(api_key=key) for key in api_keys]
        # Sticky across calls (not reset per call), so once an early key is exhausted,
        # later calls start from wherever rotation left off instead of hammering it again.
        self._current = 0

    def chat_completion(self, **kwargs):
        """Create a chat completion, rotating to the next key on a 429 rate-limit."""
        last_error: RateLimitError | None = None
        for _ in range(len(self._clients)):
            client = self._clients[self._current]
            try:
                return client.chat.completions.create(**kwargs)
            except RateLimitError as exc:
                logger.warning("Groq key #%d rate-limited, rotating to next key", self._current)
                last_error = exc
                self._current = (self._current + 1) % len(self._clients)

        raise NoAvailableGroqKeyError(f"All {len(self._clients)} configured Groq API keys are rate-limited") from last_error


def build_groq_client(settings: Settings | None = None) -> RotatingGroqClient:
    settings = settings or get_settings()
    return RotatingGroqClient(settings.groq_api_keys)
