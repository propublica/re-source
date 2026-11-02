"""Comparison logic for retrieval snapshots."""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "capture_baseline", Path(__file__).resolve().parent.parent / "scripts" / "capture_baseline.py"
)
cb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cb)


def _ranked(*ids):
    return [{"parent_id": i, "score": 1.0} for i in ids]


# ----- overlap ---------------------------------------------------------------


def test_identical_lists_overlap_fully():
    r = _ranked("a", "b", "c")
    assert cb.overlap_at(r, r, 10) == 1.0


def test_disjoint_lists_have_zero_overlap():
    assert cb.overlap_at(_ranked("a", "b"), _ranked("c", "d"), 10) == 0.0


def test_overlap_is_jaccard_over_the_union():
    # {a,b,c} vs {b,c,d}: 2 shared of 4 distinct.
    assert cb.overlap_at(_ranked("a", "b", "c"), _ranked("b", "c", "d"), 10) == 0.5


def test_overlap_respects_the_k_cutoff():
    a = _ranked("a", "b", "c")
    b = _ranked("a", "b", "zzz", "c")
    assert cb.overlap_at(a, b, 2) == 1.0


def test_two_empty_lists_are_none_not_one():
    assert cb.overlap_at([], [], 10) is None


def test_empty_against_nonempty_is_zero():
    assert cb.overlap_at([], _ranked("a"), 10) == 0.0


# ----- displacement ----------------------------------------------------------


def test_displacement_is_zero_when_order_is_unchanged():
    r = _ranked("a", "b", "c")
    d = cb.rank_displacement(r, r)
    assert d == {"shared": 3, "median_move": 0.0, "max_move": 0}


def test_displacement_measures_how_far_shared_ids_moved():
    d = cb.rank_displacement(_ranked("a", "b", "c"), _ranked("c", "b", "a"))
    assert d["shared"] == 3
    assert d["max_move"] == 2


def test_displacement_is_none_without_shared_ids():
    assert cb.rank_displacement(_ranked("a"), _ranked("b")) is None


# ----- queries file ----------------------------------------------------------


def test_load_queries_skips_blanks_and_comments(tmp_path):
    f = tmp_path / "q.txt"
    f.write_text("# a comment\n\nteachers\n\n  unsafe classrooms  \n", encoding="utf-8")
    assert cb.load_queries(f) == ["teachers", "unsafe classrooms"]


def test_load_queries_accepts_the_batch_search_pipe_form(tmp_path):
    f = tmp_path / "q.txt"
    f.write_text("methane_tips|methane emissions\nplain query\n", encoding="utf-8")
    assert cb.load_queries(f) == ["methane emissions", "plain query"]


# ----- score stats -----------------------------------------------------------


def test_score_stats_reports_the_spread():
    s = cb.score_stats([1.0, 5.0, 3.0])
    assert s == {"min": 1.0, "median": 3.0, "max": 5.0, "count": 3}


def test_score_stats_is_none_when_empty():
    assert cb.score_stats([]) is None


# ----- no tip text leaks into a snapshot -------------------------------------


def test_ranked_entries_record_no_document_text():

    class _R:
        doc_id = "tips__recABC"
        score = 12.5
        text = "SECRET tip body that must not be recorded"
        fields = {"Email": "someone@example.com"}
        payload = {"lanes": ["atomic"], "matching_chunks": [{"text": "also secret"}]}

    entries = cb.ranked_entries([_R()], top_n=10)
    assert entries == [{"parent_id": "tips__recABC", "score": 12.5, "lanes": ["atomic"]}]
    blob = repr(entries)
    assert "SECRET" not in blob and "example.com" not in blob and "also secret" not in blob


def test_ranked_entries_truncates_to_top_n():
    class _R:
        def __init__(self, i):
            self.doc_id, self.score, self.payload = f"p{i}", float(i), None

    assert len(cb.ranked_entries([_R(i) for i in range(100)], top_n=5)) == 5


# ----- csv query sets --------------------------------------------------------


def test_load_queries_reads_a_csv_query_column(tmp_path):
    f = tmp_path / "queries.csv"
    f.write_text("output_name,query\nmethane_tips,methane emissions\nteachers,unsafe classrooms\n",
                 encoding="utf-8")
    assert cb.load_queries(f) == ["methane emissions", "unsafe classrooms"]


def test_load_queries_accepts_search_as_the_column_name(tmp_path):
    f = tmp_path / "queries.csv"
    f.write_text("name,search\na,methane emissions\n", encoding="utf-8")
    assert cb.load_queries(f) == ["methane emissions"]


def test_load_queries_column_match_is_case_insensitive(tmp_path):
    f = tmp_path / "queries.csv"
    f.write_text("Output_Name,Query\na,methane emissions\n", encoding="utf-8")
    assert cb.load_queries(f) == ["methane emissions"]


def test_load_queries_falls_back_to_the_last_column(tmp_path):
    f = tmp_path / "queries.csv"
    f.write_text("label,text\nmethane_tips,methane emissions\n", encoding="utf-8")
    assert cb.load_queries(f) == ["methane emissions"]


def test_load_queries_skips_blank_csv_rows(tmp_path):
    f = tmp_path / "queries.csv"
    f.write_text("query\nmethane emissions\n\n   \nunsafe classrooms\n", encoding="utf-8")
    assert cb.load_queries(f) == ["methane emissions", "unsafe classrooms"]


def test_load_queries_tolerates_a_bom(tmp_path):
    f = tmp_path / "queries.csv"
    f.write_bytes("query\nmethane emissions\n".encode("utf-8-sig"))
    assert cb.load_queries(f) == ["methane emissions"]


def test_a_capture_measures_the_depth_the_app_actually_runs():
    import inspect

    from re_source.search_runner import SearchRun
    from re_source.vespa_strategies import DEFAULT_PAGE_SIZE, DEFAULT_TOP_K

    assert cb.DEFAULT_TOP_K is DEFAULT_TOP_K
    assert inspect.signature(SearchRun.first_page).parameters["top_k"].default is DEFAULT_TOP_K
    assert DEFAULT_PAGE_SIZE == 50
