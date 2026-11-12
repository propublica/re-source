"""Union logic for the batch Author Profile export."""

import importlib.util
from pathlib import Path

from re_source.search_types import SearchResult

_spec = importlib.util.spec_from_file_location(
    "author_batch_export",
    Path(__file__).resolve().parent.parent / "scripts" / "author_batch_export.py",
)
abe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(abe)


def _result(doc_id, score, field, *, reason=None):
    return SearchResult(
        doc_id=doc_id,
        text="",
        source_file="appAbC123DeF456Gh",
        fields={"_record_id": doc_id},
        score=score,
        payload={
            "matching_chunks": [],
            "evidence": [{"field": field, "quote": f"I am a {field}"}],
            "llm_reason": reason,
        },
    )


def test_a_record_found_once_keeps_its_query_label():
    merged = abe.merge_results([("teachers", [_result("a", 0.5, "Job")])])
    assert [r.doc_id for r in merged] == ["a"]
    assert merged[0].payload["queries"] == ["teachers"]


def test_a_record_found_twice_appears_once_naming_both_queries():
    merged = abe.merge_results([
        ("teachers", [_result("a", 0.5, "Job")]),
        ("school staff", [_result("a", 0.4, "Job")]),
    ])
    assert len(merged) == 1
    assert merged[0].payload["queries"] == ["teachers", "school staff"]


def test_evidence_from_every_query_is_unioned():
    merged = abe.merge_results([
        ("teachers", [_result("a", 0.5, "Job")]),
        ("parents", [_result("a", 0.4, "Household")]),
    ])
    fields = [e["field"] for e in merged[0].payload["evidence"]]
    assert fields == ["Job", "Household"]


def test_identical_evidence_is_not_duplicated():
    merged = abe.merge_results([
        ("teachers", [_result("a", 0.5, "Job")]),
        ("educators", [_result("a", 0.4, "Job")]),
    ])
    assert len(merged[0].payload["evidence"]) == 1


def test_the_best_scoring_run_supplies_score_and_reasoning():
    merged = abe.merge_results([
        ("teachers", [_result("a", 0.2, "Job", reason="weak")]),
        ("educators", [_result("a", 0.9, "Job", reason="strong")]),
    ])
    assert merged[0].score == 0.9
    assert merged[0].payload["llm_reason"] == "strong"
    # ...without losing what the earlier, lower-scoring query contributed.
    assert merged[0].payload["queries"] == ["teachers", "educators"]


def test_a_later_lower_scoring_run_does_not_overwrite_reasoning():
    merged = abe.merge_results([
        ("teachers", [_result("a", 0.9, "Job", reason="strong")]),
        ("educators", [_result("a", 0.2, "Job", reason="weak")]),
    ])
    assert merged[0].score == 0.9
    assert merged[0].payload["llm_reason"] == "strong"


def test_results_are_ranked_by_score_across_queries():
    merged = abe.merge_results([
        ("teachers", [_result("a", 0.2, "Job"), _result("b", 0.8, "Job")]),
        ("nurses", [_result("c", 0.5, "Job")]),
    ])
    assert [r.doc_id for r in merged] == ["b", "c", "a"]


def test_no_queries_produces_no_rows():
    assert abe.merge_results([]) == []
    assert abe.merge_results([("teachers", [])]) == []


# ----- the export door -------------------------------------------------------
#
# The search half still needs a live Vespa, but main()'s tail — write the
# workbook, record the disclosure — does not, and it had no coverage at all.
# It is the half that decides whether records can leave unaccounted for.

import openpyxl
import pytest

from re_source import audit


@pytest.fixture
def recorded(monkeypatch):
    calls = []
    monkeypatch.setattr(audit, "log_event", lambda *a, **kw: calls.append((a, kw)))
    return calls


def _run_main(monkeypatch, tmp_path, results, *, queries="teachers|tips from teachers"):
    """Drive main() with the Vespa/LLM half stubbed out."""
    queries_file = tmp_path / "queries.txt"
    queries_file.write_text(queries + "\n")
    out = tmp_path / "sources.xlsx"

    class _Result:
        def __init__(self, rs):
            self.results = rs
            self.metadata = {"before_llm": len(rs)}

    class _Strategy:
        def execute(self, **_):
            return _Result(results)

    monkeypatch.setattr(abe, "build_strategy", lambda name: _Strategy())
    monkeypatch.setattr(
        abe.sys, "argv",
        ["author_batch_export.py", "-f", str(queries_file), "-o", str(out),
         "--actor", "batch@example.com"],
    )
    return out, abe.main()


def test_main_records_one_export_event(monkeypatch, tmp_path, recorded):
    out, rc = _run_main(monkeypatch, tmp_path, [_result("a", 0.5, "Job")])

    assert rc == 0
    (args, kwargs), = recorded
    assert args[0] == audit.EVENT_EXPORT
    assert kwargs["surface"] == audit.SURFACE_CLI
    assert kwargs["actor"] == "batch@example.com"
    assert kwargs["strategy"] == abe.STRATEGY


def test_main_records_every_record_in_the_workbook(monkeypatch, tmp_path, recorded):
    _run_main(
        monkeypatch, tmp_path,
        [_result("a", 0.5, "Job"), _result("b", 0.4, "Job")],
        queries="teachers|tips from teachers\nnurses|nurses and hospital staff",
    )

    (_, kwargs), = recorded
    assert sorted(kwargs["record_ids"]) == [
        "appAbC123DeF456Gh::a", "appAbC123DeF456Gh::b"
    ]


def test_main_records_the_file_it_actually_wrote(monkeypatch, tmp_path, recorded):
    out, _ = _run_main(monkeypatch, tmp_path, [_result("a", 0.5, "Job")])

    (_, kwargs), = recorded
    assert kwargs["filename"] == out.name
    assert openpyxl.load_workbook(out).sheetnames[0] == "Summary"


def test_main_leaves_the_queries_out_of_the_audit_log(monkeypatch, tmp_path, recorded):
    _run_main(
        monkeypatch, tmp_path, [_result("a", 0.5, "Job")],
        queries="teachers|tips from teachers\nnurses|nurses and hospital staff",
    )

    (_, kwargs), = recorded
    assert "query" not in kwargs


def test_nothing_to_export_records_nothing(monkeypatch, tmp_path, recorded):
    out, rc = _run_main(monkeypatch, tmp_path, [])

    assert rc == 0
    assert recorded == []
    assert not out.exists()


# ----- run tuning --------------------------------------------------------------


class _FakeResult:
    def __init__(self, results):
        self.results = results
        self.metadata = {"before_llm": len(results)}


class _FakeStrategy:
    """Answers each query with one result named after it; 'boom' raises."""

    def execute(self, *, query, top_k):
        if query == "boom":
            raise RuntimeError("rate limited")
        return _FakeResult([_result(query, 0.5, "Job")])


def test_parallel_queries_come_back_in_input_order():
    queries = [(f"q{i}", f"query {i}") for i in range(8)]

    per_query, failed = abe.run_queries(_FakeStrategy(), queries, top_k=10, parallel=4)

    assert [label for label, _ in per_query] == [f"q{i}" for i in range(8)]
    assert failed == []


def test_a_failed_query_is_reported_not_dropped_silently():
    queries = [("ok", "fine"), ("bad", "boom"), ("ok2", "also fine")]

    per_query, failed = abe.run_queries(_FakeStrategy(), queries, top_k=10, parallel=2)

    assert [label for label, _ in per_query] == ["ok", "ok2"]
    assert failed == ["bad"]


def test_batching_overrides_apply_to_this_run_only(monkeypatch):
    from re_source.vespa_strategies import AuthorProfileLLMVespaStrategy

    monkeypatch.setattr(abe, "build_strategy",
                        lambda name: AuthorProfileLLMVespaStrategy(client=None))

    tuned = abe.build_tuned_strategy(batch_size=5, workers=20)

    assert (tuned.VALIDATION_BATCH_SIZE, tuned.VALIDATION_WORKERS) == (5, 20)
    assert AuthorProfileLLMVespaStrategy.VALIDATION_BATCH_SIZE == 12
    assert AuthorProfileLLMVespaStrategy.VALIDATION_WORKERS == 8


def test_no_overrides_keeps_the_strategy_defaults(monkeypatch):
    from re_source.vespa_strategies import AuthorProfileLLMVespaStrategy

    monkeypatch.setattr(abe, "build_strategy",
                        lambda name: AuthorProfileLLMVespaStrategy(client=None))

    tuned = abe.build_tuned_strategy(batch_size=None, workers=None)

    assert (tuned.VALIDATION_BATCH_SIZE, tuned.VALIDATION_WORKERS) == (12, 8)


# ----- one tab per query --------------------------------------------------------


def _run_tabbed(monkeypatch, tmp_path, by_query: dict):
    """Drive main() with --tab-per-query; each query returns its own results."""
    queries_file = tmp_path / "queries.txt"
    queries_file.write_text("".join(f"{label}|{label}\n" for label in by_query))
    out = tmp_path / "tabs.xlsx"

    class _Strategy:
        def execute(self, *, query, top_k):
            return _FakeResult(by_query[query]())

    monkeypatch.setattr(abe, "build_strategy", lambda name: _Strategy())
    monkeypatch.setattr(
        abe.sys, "argv",
        ["author_batch_export.py", "-f", str(queries_file), "-o", str(out),
         "--tab-per-query", "--actor", "batch@example.com"],
    )
    return out, abe.main()


def test_each_tab_keeps_its_own_querys_reason(monkeypatch, tmp_path, recorded):
    # One person kept by two queries: merging promotes the higher-scoring reason,
    # which must not leak onto the other query's tab.
    out, rc = _run_tabbed(monkeypatch, tmp_path, {
        "Walmart": lambda: [_result("a", 0.4, "Job", reason="says Walmart")],
        "Target": lambda: [_result("a", 0.9, "Job", reason="says Target")],
    })

    assert rc == 0
    wb = openpyxl.load_workbook(out)
    assert wb.sheetnames == ["Summary", "Walmart", "Target"]
    assert wb["Walmart"].cell(row=3, column=2).value == "says Walmart"
    assert wb["Target"].cell(row=3, column=2).value == "says Target"


def test_tabbed_export_records_one_event_for_every_record(monkeypatch, tmp_path, recorded):
    _run_tabbed(monkeypatch, tmp_path, {
        "Walmart": lambda: [_result("a", 0.4, "Job"), _result("b", 0.3, "Job")],
        "Target": lambda: [_result("a", 0.9, "Job")],
    })

    (args, kwargs), = recorded
    assert args[0] == audit.EVENT_EXPORT
    assert sorted(kwargs["record_ids"]) == [
        "appAbC123DeF456Gh::a", "appAbC123DeF456Gh::b"
    ]
