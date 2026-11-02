"""LLM validation batches (run in a ThreadPoolExecutor) must trace under the parent span,
not spawn their own root traces.
"""

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from re_source.llm_utils import run_llm_validation


def test_threaded_validation_stays_in_parent_trace():
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")

    seen_trace_ids = []

    def fake_validate(query, texts, max_doc_chars=1000):
        # Emit a child span the way @observe would, and record its trace id.
        with tracer.start_as_current_span("validate_batch"):
            seen_trace_ids.append(trace.get_current_span().get_span_context().trace_id)
        return [{"relevant": True, "reason": ""} for _ in texts]

    items = list(range(6))  # 3 batches of 2
    with tracer.start_as_current_span("search") as parent:
        parent_trace_id = parent.get_span_context().trace_id
        run_llm_validation(
            "q", items, text_fn=lambda xs: [str(x) for x in xs],
            batch_size=2, max_workers=3, validate_fn=fake_validate,
        )

    assert len(seen_trace_ids) == 3
    # Every worker-thread batch shares the parent's trace id (nested, not stray).
    assert all(tid == parent_trace_id for tid in seen_trace_ids)


def test_content_lanes_run_in_parent_trace():
    from re_source.vespa_strategies import CombinedVespaStrategy, StrategyResult

    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(InMemorySpanExporter()))
    tracer = provider.get_tracer("test")

    seen_trace_ids = []

    class FakeLane:
        def execute(self, query, top_k):
            seen_trace_ids.append(trace.get_current_span().get_span_context().trace_id)
            return StrategyResult(results=[], elapsed_seconds=0.0, approach_name="fake")

    combo = CombinedVespaStrategy(client=object())
    combo._lanes = {name: FakeLane() for name in combo._lanes}  # no network
    lane_count = len(combo._lanes)

    with tracer.start_as_current_span("content") as parent:
        parent_trace_id = parent.get_span_context().trace_id
        combo.execute("teachers", top_k=5)  # all lanes empty -> early return

    # Derived, not hardcoded: this test is about trace nesting, so adding or
    # removing a lane shouldn't fail it. test_registry covers which lanes exist.
    assert lane_count > 1, "the parallel fan-out is what's under test"
    assert len(seen_trace_ids) == lane_count
    assert all(tid == parent_trace_id for tid in seen_trace_ids)
