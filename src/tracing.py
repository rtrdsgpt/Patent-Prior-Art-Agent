"""OpenTelemetry tracing for the agent pipeline (todo.md section 6): one span per pipeline
stage (disclosure-parser → search → [claims-parser, comparison] per candidate → citation
guard → risk-report), nested under one root span per pipeline run.

Exports to the console by default, not a hosted Langfuse/OTLP backend — this project has no
credentials for one, and `ConsoleSpanExporter` is real, inspectable tracing on its own, not a
placeholder. `opentelemetry-exporter-otlp-proto-grpc` is already in the dependency tree
(pulled in transitively by other packages), so pointing `_build_provider` at a real backend
later is a one-line swap, not a rewrite.
"""

from __future__ import annotations

import functools
from typing import Callable, TypeVar

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from opentelemetry.trace import Status, StatusCode

_SERVICE_NAME = "patent-prior-art-agent"
_provider_initialized = False


def _ensure_provider() -> None:
    """Idempotent: the real SDK provider should only ever be installed once per process,
    but every `get_tracer()` call needs to trigger this check (there's no single obvious
    "app startup" hook shared by the API, MCP server, and CLI scripts that all use this)."""
    global _provider_initialized
    if _provider_initialized:
        return
    provider = TracerProvider(resource=Resource.create({"service.name": _SERVICE_NAME}))
    # SimpleSpanProcessor (synchronous, no background export thread), not BatchSpanProcessor
    # — a console exporter is already cheap/local, so there's no batching benefit, and
    # BatchSpanProcessor's background thread can outlive a test session's captured stdout
    # and crash trying to write to it after close (hit this for real running the suite).
    provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)
    _provider_initialized = True


def get_tracer() -> trace.Tracer:
    _ensure_provider()
    return trace.get_tracer(_SERVICE_NAME)


F = TypeVar("F", bound=Callable)


def traced(span_name: str) -> Callable[[F], F]:
    """Decorator: run the wrapped function inside a span named `span_name`. Marks the span
    OK on success or records the exception and marks it ERROR on failure — always
    re-raises, this never swallows an error to make a span look clean."""

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            tracer = get_tracer()
            with tracer.start_as_current_span(span_name) as span:
                try:
                    result = fn(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as exc:
                    span.record_exception(exc)
                    span.set_status(Status(StatusCode.ERROR, str(exc)))
                    raise

        return wrapper

    return decorator
