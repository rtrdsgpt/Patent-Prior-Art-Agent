import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from tracing import traced


@pytest.fixture
def exporter(monkeypatch):
    # A fresh, test-local TracerProvider + InMemorySpanExporter rather than touching the
    # real module-level provider (`tracing._ensure_provider()` installs a real, global,
    # effectively-once-only OpenTelemetry provider — swapping it mid-test-suite isn't
    # something the API supports cleanly). Monkeypatching `traced`'s `get_tracer` call is
    # what lets these tests assert on actually-recorded spans without touching that
    # process-wide singleton at all.
    provider = TracerProvider()
    memory_exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(memory_exporter))
    tracer = provider.get_tracer("test")
    monkeypatch.setattr("tracing.get_tracer", lambda: tracer)
    return memory_exporter


def test_traced_returns_function_result(exporter):
    @traced("my_span")
    def fn(x):
        return x * 2

    assert fn(3) == 6


def test_traced_passes_through_args_and_kwargs(exporter):
    @traced("my_span")
    def fn(a, b, c=3):
        return a + b + c

    assert fn(1, 2, c=10) == 13


def test_traced_creates_a_span_with_the_given_name(exporter):
    @traced("my_span")
    def fn():
        return 1

    fn()
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "my_span"


def test_traced_marks_span_ok_on_success(exporter):
    @traced("my_span")
    def fn():
        return 1

    fn()
    assert exporter.get_finished_spans()[0].status.status_code == StatusCode.OK


def test_traced_marks_span_error_and_reraises_on_exception(exporter):
    @traced("my_span")
    def fn():
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        fn()

    assert exporter.get_finished_spans()[0].status.status_code == StatusCode.ERROR


def test_traced_records_exception_as_a_span_event(exporter):
    @traced("my_span")
    def fn():
        raise ValueError("boom")

    with pytest.raises(ValueError):
        fn()

    span = exporter.get_finished_spans()[0]
    assert any(event.name == "exception" for event in span.events)


def test_nested_traced_spans_are_parent_child(exporter):
    @traced("child")
    def child():
        return 1

    @traced("parent")
    def parent():
        return child()

    parent()
    spans = {s.name: s for s in exporter.get_finished_spans()}
    assert spans["child"].parent.span_id == spans["parent"].context.span_id
