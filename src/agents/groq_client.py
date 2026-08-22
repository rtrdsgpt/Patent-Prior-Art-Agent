"""Rotating Groq chat model: cycles across multiple API keys on a rate limit.

Built on `langchain_groq.ChatGroq` (LangChain's LLM layer), not the raw `groq` SDK directly
— every agent now goes through LangChain's `invoke()`/`with_structured_output()` interface.
LangChain has no built-in multi-key rotation, so `RotatingChatGroq` wraps one `ChatGroq`
instance per key and re-implements the same rotation this project already had with the raw
SDK: Groq's free tier has fairly tight per-key rate limits, and the agent pipeline makes
several LLM calls per job across multiple stages — a single free-tier key would throttle
that quickly. The keys provided for this project are multiple free-tier keys specifically
for that reason.

`ChatGroq` still calls the same underlying `groq` SDK client under the hood (confirmed by
tracing a live call's traceback through `langchain_groq/chat_models.py` into
`groq/_base_client.py`), so it still raises `groq.RateLimitError` on a 429 — rotation
catches exactly that, not a LangChain-specific exception type.

Bounded to at most one attempt per configured key, same bounded-retry discipline as todo.md's
note to "reuse the bounded-retry pattern from Exporter Crawl's discovery.py" for the search
agent — retry with a clear stopping point, never loop indefinitely.
"""

from __future__ import annotations

import logging
from typing import Callable, TypeVar

from groq import RateLimitError
from langchain_groq import ChatGroq

from config.settings import Settings, get_settings

logger = logging.getLogger(__name__)

T = TypeVar("T")


class NoAvailableGroqKeyError(RuntimeError):
    """Raised when every configured Groq API key is rate-limited."""


class RotatingChatGroq:
    def __init__(self, api_keys: list[str], model: str, temperature: float = 0) -> None:
        if not api_keys:
            raise ValueError("RotatingChatGroq requires at least one API key")
        self._models = [ChatGroq(model=model, api_key=key, temperature=temperature) for key in api_keys]
        # Sticky across calls (not reset per call), so once an early key is exhausted,
        # later calls start from wherever rotation left off instead of hammering it again.
        self._current = 0

    def invoke(self, messages: list[dict]):
        """Plain chat completion — returns a LangChain `AIMessage` (`.content` is the text)."""
        return self._rotate(lambda model: model.invoke(messages))

    def with_structured_output(self, schema: type) -> "_RotatingStructuredOutput":
        """Mirrors `ChatGroq.with_structured_output()` — returns a runnable whose own
        `.invoke()` rotates keys the same way `invoke()` above does."""
        return _RotatingStructuredOutput(self, schema)

    def _rotate(self, operation: Callable[[ChatGroq], T]) -> T:
        last_error: RateLimitError | None = None
        for _ in range(len(self._models)):
            model = self._models[self._current]
            try:
                return operation(model)
            except RateLimitError as exc:
                logger.warning("Groq key #%d rate-limited, rotating to next key", self._current)
                last_error = exc
                self._current = (self._current + 1) % len(self._models)

        raise NoAvailableGroqKeyError(f"All {len(self._models)} configured Groq API keys are rate-limited") from last_error


class _RotatingStructuredOutput:
    """The structured-output counterpart of `RotatingChatGroq` — `with_structured_output()`
    on a plain `ChatGroq` returns a new runnable per call, so this defers building that
    runnable until `.invoke()` actually picks which underlying model to use.

    Uses `method="json_schema"`, not LangChain's default `"function_calling"` (forced
    tool-choice) — found this the hard way running a real multi-claim patent through the
    live pipeline: the default method failed with `groq.BadRequestError: Tool choice is
    required, but model did not call a tool` (the model narrated a nicely formatted answer
    in prose instead of invoking the tool). Reproduced the exact failure standalone, then
    tried `method="json_schema"` against the same prompt and it succeeded — a live,
    verified fix, not a guess from the docs.
    """

    def __init__(self, rotating_client: RotatingChatGroq, schema: type) -> None:
        self._rotating_client = rotating_client
        self._schema = schema

    def invoke(self, messages: list[dict]):
        return self._rotating_client._rotate(lambda model: model.with_structured_output(self._schema, method="json_schema").invoke(messages))


def build_groq_client(settings: Settings | None = None) -> RotatingChatGroq:
    settings = settings or get_settings()
    return RotatingChatGroq(settings.groq_api_keys, model=settings.groq_model)
