"""One module pairs the workbook with the event that accounts for it."""

import re
from io import BytesIO
from pathlib import Path

import openpyxl
import pytest

from re_source import audit, disclosure
from re_source.search_types import SearchResult


def _result(rec="recA", source="appAbC123DeF456Gh"):
    return SearchResult(
        doc_id=f"doc-{rec}",
        text="mold everywhere",
        source_file=source,
        fields={"_record_id": rec, "Plain": "just text"},
        score=0.87,
        payload={"matching_chunks": [{"column_name": "What happened?",
                                      "text": "mold everywhere"}]},
    )


@pytest.fixture
def recorded(monkeypatch):
    """Capture log_event calls instead of writing them anywhere."""
    calls = []
    monkeypatch.setattr(
        audit, "log_event", lambda *a, **kw: calls.append((a, kw)) or {"event_id": "x"}
    )
    return calls


# ----- preparing -----------------------------------------------------------

def test_prepared_export_carries_the_rows_it_will_record():
    d = disclosure.prepare_export([_result("recA"), _result("recB")], actor="a@x.com")
    assert d.record_ids == ["appAbC123DeF456Gh::recA", "appAbC123DeF456Gh::recB"]


def test_prepared_export_holds_a_real_workbook():
    d = disclosure.prepare_export([_result()], actor="a@x.com")
    wb = openpyxl.load_workbook(BytesIO(d.data))
    assert "Summary" in wb.sheetnames


def test_preparing_does_not_record(recorded):
    disclosure.prepare_export([_result()], actor="a@x.com")
    assert recorded == []


# ----- recording -----------------------------------------------------------

def test_record_logs_one_export_event(recorded):
    d = disclosure.prepare_export(
        [_result()], actor="dana@example.com", surface=audit.SURFACE_STREAMLIT,
        strategy="Content", session_id="sess-1",
    )
    d.record()

    (args, kwargs), = recorded
    assert args[0] == audit.EVENT_EXPORT
    assert kwargs["actor"] == "dana@example.com"
    assert kwargs["surface"] == "streamlit"
    assert "query" not in kwargs
    assert kwargs["strategy"] == "Content"
    assert kwargs["session_id"] == "sess-1"
    assert kwargs["filename"] == d.filename


def test_record_passes_through_what_log_event_returned(monkeypatch):
    monkeypatch.setattr(audit, "log_event", lambda *a, **kw: None)
    assert disclosure.prepare_export([_result()], actor="a").record() is None

    monkeypatch.setattr(audit, "log_event", lambda *a, **kw: {"event_id": "abc"})
    assert disclosure.prepare_export([_result()], actor="a").record() == {"event_id": "abc"}


def test_recording_twice_records_twice(recorded):
    d = disclosure.prepare_export([_result()], actor="a@x.com")
    d.record()
    d.record()
    assert len(recorded) == 2


# ----- writing -------------------------------------------------------------

def test_write_puts_the_workbook_on_disk(tmp_path, recorded):
    out = tmp_path / "workplace_mentions.xlsx"
    disclosure.prepare_export([_result()], actor="a@x.com").write(out)

    assert openpyxl.load_workbook(out).sheetnames[0] == "Summary"


def test_write_records_the_filename_it_wrote(tmp_path, recorded):
    out = tmp_path / "workplace_mentions.xlsx"
    disclosure.prepare_export([_result()], actor="a@x.com").write(out)

    (_, kwargs), = recorded
    assert kwargs["filename"] == "workplace_mentions.xlsx"


def test_write_records_exactly_once(tmp_path, recorded):
    disclosure.prepare_export([_result()], actor="a@x.com").write(tmp_path / "o.xlsx")
    assert len(recorded) == 1


# ----- naming --------------------------------------------------------------

def test_filename_is_a_timestamp_when_not_given():
    d = disclosure.prepare_export([_result()], actor="a")
    assert re.fullmatch(r"resource_\d{8}_\d{6}\.xlsx", d.filename)


def test_an_explicit_filename_wins():
    d = disclosure.prepare_export([_result()], actor="a", filename="given.xlsx")
    assert d.filename == "given.xlsx"


# ----- actor ---------------------------------------------------------------

def test_default_actor_prefers_the_environment_over_the_os_user(monkeypatch):
    monkeypatch.setenv("AUDIT_ACTOR", "batch-runner@example.com")
    assert disclosure.default_actor() == "batch-runner@example.com"


def test_default_actor_falls_back_to_the_os_user(monkeypatch):
    monkeypatch.delenv("AUDIT_ACTOR", raising=False)
    assert disclosure.default_actor()  # whatever it is, it isn't empty


def test_an_empty_audit_actor_does_not_win(monkeypatch):
    monkeypatch.setenv("AUDIT_ACTOR", "")
    assert disclosure.default_actor() != ""


# ----- the surface default -------------------------------------------------

def test_the_default_surface_is_the_offline_one(recorded):
    disclosure.prepare_export([_result()], actor="a").record()
    (_, kwargs), = recorded
    assert kwargs["surface"] == audit.SURFACE_CLI


def test_an_empty_result_set_still_prepares(recorded):
    d = disclosure.prepare_export([], actor="a")
    assert d.record_ids == []
