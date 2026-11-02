"""What an upload would do, and in what order it is decided."""

import pandas as pd
import pytest

import re_source.datasets as datasets
import re_source.import_plan as import_plan_module
from re_source.import_plan import (
    ImportPlan,
    _friendly_db_error,
    column_change_note,
    column_delta,
    confirm_label,
    content_delta,
    describe_content_delta,
    id_column_candidates,
    plan,
    preflight_warnings,
    replace_notice,
    suggest_table_name,
    uploadable_tables,
)


@pytest.fixture(autouse=True)
def synced(monkeypatch):
    """A table shape the deployment has told us a sync owns."""
    monkeypatch.setattr(datasets, "READ_ONLY_TABLE_PATTERNS", (r"^ext[A-Za-z0-9]{14}$",))


class _FakeDB:
    """A stand-in for the project's DuckDB, counting what it was asked."""

    def __init__(self, tables=None, rows=None, hashes=None, columns=None,
                 ids=None, raises=None, hidden=None):
        self.tables = tables if tables is not None else []
        # Tables present in the database but absent from list_tables — the
        # caches the project's own scripts own.
        self.hidden = hidden if hidden is not None else []
        self.rows = rows or {}
        self.hashes = hashes or {}
        self.columns = columns or {}
        self.ids = ids or {}
        self.raises = raises or set()
        self.reads = []

    def list_tables(self):
        self.reads.append("list_tables")
        return list(self.tables)

    def all_tables(self):
        self.reads.append("all_tables")
        return sorted({*self.tables, *self.hidden})

    def record_ids(self, table):
        self.reads.append(f"record_ids:{table}")
        return set(self.ids.get(table, ()))

    def count_rows(self, table):
        self.reads.append(f"count_rows:{table}")
        if "count_rows" in self.raises:
            raise RuntimeError("no")
        return self.rows.get(table, 0)

    def content_hashes(self, table):
        self.reads.append(f"content_hashes:{table}")
        if "content_hashes" in self.raises:
            raise RuntimeError("no")
        return dict(self.hashes.get(table, {}))

    def column_names(self, table):
        self.reads.append(f"column_names:{table}")
        return list(self.columns.get(table, []))


def _df(ids=("r1", "r2"), **cols):
    data = {"id": list(ids)}
    data.update({k: list(v) for k, v in cols.items()})
    return pd.DataFrame(data)


def _plan(df=None, *, table="tips", id_column="id", db=None):
    return plan(df if df is not None else _df(), id_column=id_column,
                table=table, db=db or _FakeDB())


# ----- creating a new table --------------------------------------------------

def test_a_new_table_is_not_a_replace():
    p = _plan()
    assert p.is_replace is False
    assert p.needs_confirmation is False


def test_a_new_table_may_proceed_without_a_confirmation():
    assert _plan().may_proceed(confirmed=False) is True


def test_a_new_table_has_nothing_to_say_about_changes():
    p = _plan()
    assert p.notice is None
    assert p.change_summary is None


# ----- replacing an existing one ---------------------------------------------

def _replace_db(**kw):
    return _FakeDB(
        tables=["tips"],
        rows={"tips": 2},
        hashes={"tips": {"tips::r1": "h1", "tips::r2": "h2"}},
        columns={"tips": ["_record_id", "note"]},
        **kw,
    )


def test_replacing_an_existing_table_needs_a_confirmation():
    p = _plan(db=_replace_db())
    assert p.is_replace is True
    assert p.needs_confirmation is True


def test_a_replace_does_not_proceed_unconfirmed():
    assert _plan(db=_replace_db()).may_proceed(confirmed=False) is False


def test_a_replace_proceeds_once_confirmed():
    assert _plan(db=_replace_db()).may_proceed(confirmed=True) is True


def test_a_replace_says_what_is_there_now():
    assert "2 rows" in _plan(db=_replace_db()).notice


def test_a_replace_reports_what_would_change():
    p = _plan(db=_replace_db())
    assert p.change_summary
    assert p.content is not None


# ----- the gate that guards a synced table ------------------------------------

def test_a_synced_table_is_refused_as_a_destination():
    p = _plan(table="extAbC123DeF456Gh", db=_FakeDB(tables=["extAbC123DeF456Gh"]))

    assert p.blocked is True
    assert any(level == "error" for level, _ in p.problems)


def test_a_refused_destination_offers_no_confirmation_to_tick():
    p = _plan(table="extAbC123DeF456Gh", db=_FakeDB(tables=["extAbC123DeF456Gh"]))

    assert p.needs_confirmation is False
    assert p.may_proceed(confirmed=True) is False


def test_a_refused_destination_is_never_read_from():
    db = _FakeDB(tables=["extAbC123DeF456Gh"])
    _plan(table="extAbC123DeF456Gh", db=db)

    assert not any(r.startswith(("content_hashes", "count_rows")) for r in db.reads)


def test_no_error_ever_proceeds_however_the_box_is_ticked():
    p = ImportPlan(table="t", problems=(("error", "nope"),))
    assert p.may_proceed(confirmed=True) is False


def test_a_warning_does_not_block():
    p = ImportPlan(table="t", problems=(("warning", "heads up"),))
    assert p.blocked is False
    assert p.may_proceed(confirmed=False) is True


# ----- what it costs to decide -----------------------------------------------

def test_the_collision_scan_spans_every_other_table(monkeypatch):
    monkeypatch.setattr(import_plan_module.config, "DOCUMENT_ID_SCHEME", "legacy")
    db = _FakeDB(tables=["a", "b", "tips"])
    _plan(table="tips", db=db)

    assert "record_ids:a" in db.reads
    assert "record_ids:b" in db.reads
    assert "record_ids:tips" not in db.reads      # not against itself


def test_dataset_scoped_ids_do_not_scan_other_tables(monkeypatch):
    monkeypatch.setattr(
        import_plan_module.config, "DOCUMENT_ID_SCHEME", "dataset_scoped"
    )
    db = _FakeDB(tables=["a", "tips"], ids={"a": {"r1"}})

    result = plan(_df(ids=("r1",)), id_column="id", table="tips", db=db)

    assert result.blocked is False
    assert not any(read.startswith("record_ids:") for read in db.reads)


def test_deciding_reads_each_table_once(monkeypatch):
    monkeypatch.setattr(import_plan_module.config, "DOCUMENT_ID_SCHEME", "legacy")
    db = _FakeDB(tables=["a", "b", "tips"])
    _plan(table="tips", db=db)

    assert db.reads.count("record_ids:a") == 1
    assert db.reads.count("list_tables") == 1
    assert db.reads.count("all_tables") == 1


# ----- degrading rather than blocking ----------------------------------------

def test_an_uncomparable_table_still_allows_the_import():
    db = _replace_db(raises={"content_hashes"})
    p = _plan(db=db)

    assert p.comparable is False
    assert p.may_proceed(confirmed=True) is True


def test_an_uncomparable_table_says_so_rather_than_claiming_no_changes():
    p = _plan(db=_replace_db(raises={"content_hashes"}))
    assert "unknown" in p.change_summary


def test_an_uncountable_table_is_still_a_replace():
    p = _plan(db=_replace_db(raises={"count_rows"}))
    assert p.is_replace is True
    assert p.existing_rows is None
    assert p.notice


# ----- the confirmation wording ----------------------------------------------

def test_the_confirmation_names_the_table():
    assert "tips" in _plan(db=_replace_db()).confirmation


# ----- table name / column ordering ----------------------------------------

@pytest.mark.parametrize(
    "filename,expected",
    [
        ("Housing 2026.csv", "Housing_2026"),
        ("tips-export.xlsx", "tips_export"),
        ("2026 responses.csv", "t_2026_responses"),   # no leading digit
        ("weird!!!.csv", "weird"),
    ],
)
def test_suggest_table_name_matches_the_cli_rule(filename, expected):
    assert suggest_table_name(filename) == expected


def test_id_column_candidates_puts_likely_ids_first():
    cols = ["What happened?", "Email", "Record ID", "Name"]
    assert id_column_candidates(cols)[0] == "Record ID"


def test_id_column_candidates_keeps_every_column():
    cols = ["a", "b", "Record ID", "c"]
    assert sorted(id_column_candidates(cols)) == sorted(cols)


# ----- preflight rules ------------------------------------------------------

def _frame(**cols):
    """A parsed upload, built column by column."""
    return pd.DataFrame(cols)


def _levels(problems):
    return {level for level, _ in problems}


def test_clean_file_has_no_problems():
    df = _frame(**{"Record ID": ["r1", "r2"], "What happened?": ["mold", "no heat"]})
    assert preflight_warnings(df, "Record ID", "tips", {}) == []


def test_missing_id_column_is_an_error():
    df = _frame(a=["1"])
    problems = preflight_warnings(df, "Record ID", "tips", {})
    assert _levels(problems) == {"error"}


def test_existing_record_id_column_is_an_error():
    df = _frame(**{"Record ID": ["r1"], "_record_id": ["stale"]})
    problems = preflight_warnings(df, "Record ID", "tips", {})
    assert any("_record_id" in msg for level, msg in problems if level == "error")


def test_a_stray_id_column_is_a_warning_not_a_refusal():
    """It was an error on the old precedence, where `id` beat `_record_id`.
    The picked column wins now, so refusing the import was blocking uploads
    that behave correctly."""
    df = _frame(**{"Record ID": ["r1"], "id": ["something-else"], "note": ["a"]})
    problems = preflight_warnings(df, "Record ID", "tips", {})

    assert _levels(problems) == {"warning"}
    (msg,) = [m for level, m in problems if level == "warning"]
    assert "ordinary content" in msg


def test_a_stray_id_column_does_not_block_the_import():
    """The property that regressed: this upload is fine and must proceed."""
    db = _FakeDB(tables=["tips"])
    df = pd.DataFrame({"Record ID": ["r1"], "id": ["x"], "note": ["a"]})
    p = plan(df, id_column="Record ID", table="uploads", db=db)

    assert p.blocked is False
    assert p.may_proceed(confirmed=True) is True


def test_the_warning_describes_the_precedence_that_actually_holds():
    """Pinned against db_loader rather than against prose, so the two cannot
    drift apart again the way they just did."""
    from re_source import db_loader

    doc = db_loader.document_from_row(
        {"_record_id": "picked", "id": "stray", "note": "x"}, "tips"
    )
    assert doc.id == db_loader.make_document_id("tips", "picked")

    df = _frame(**{"Record ID": ["r1"], "id": ["stray"], "note": ["a"]})
    (msg,) = [m for lvl, m in preflight_warnings(df, "Record ID", "tips", {})
              if lvl == "warning"]
    assert "override" not in msg


def test_blank_ids_are_warned_about_with_a_count():
    df = _frame(**{"Record ID": ["r1", "", None], "note": ["a", "b", "c"]})
    problems = preflight_warnings(df, "Record ID", "tips", {})
    assert [("warning", m) for _, m in problems if "2 rows" in m]


def test_duplicate_ids_are_rejected():
    df = _frame(**{"Record ID": ["r1", "r1", "r2"], "note": ["a", "b", "c"]})
    problems = preflight_warnings(df, "Record ID", "tips", {})
    assert any(level == "error" and "more than once" in msg for level, msg in problems)


def test_ids_colliding_with_another_table_is_an_error(monkeypatch):
    monkeypatch.setattr(import_plan_module.config, "DOCUMENT_ID_SCHEME", "legacy")
    df = _frame(**{"Record ID": ["r1", "r9"], "note": ["a", "b"]})
    problems = preflight_warnings(
        df, "Record ID", "tips", {"housing": {"r9", "zzz"}}
    )
    assert any(
        level == "error" and "housing" in msg for level, msg in problems
    )


def test_ids_matching_the_table_being_replaced_are_fine():
    df = _frame(**{"Record ID": ["r1", "r9"], "note": ["a", "b"]})
    problems = preflight_warnings(
        df, "Record ID", "tips", {"tips": {"r1", "r9"}}
    )
    assert problems == []


# ----- error message shaping ------------------------------------------------

def test_lock_errors_explain_themselves():
    msg = _friendly_db_error(Exception("Could not set lock on file: Conflicting lock"))
    assert "another process" in msg


def test_other_errors_say_nothing_was_changed():
    msg = _friendly_db_error(Exception("boom"))
    assert "nothing was changed" in msg and "boom" in msg


# ----- content delta --------------------------------------------------------

def test_content_delta_splits_the_four_outcomes():
    old = {"r1": "h1", "r2": "h2", "r3": "h3", "r4": "h4"}
    new = {"r1": "h1", "r2": "CHANGED", "r3": "h3", "r9": "h9"}

    assert content_delta(old, new) == {
        "unchanged": 2,   # r1 and r3: same id, same hash
        "changed": 1,     # r2: same id, different hash
        "deleted": 1,     # r4: gone from the upload
        "inserted": 1,    # r9: new in the upload
    }


def test_content_delta_identical_is_all_unchanged():
    same = {"r1": "h1", "r2": "h2"}
    assert content_delta(same, same) == {
        "unchanged": 2, "changed": 0, "deleted": 0, "inserted": 0,
    }


def test_content_delta_new_table_is_all_inserted():
    assert content_delta({}, {"r1": "h1", "r2": "h2"}) == {
        "unchanged": 0, "changed": 0, "deleted": 0, "inserted": 2,
    }


def test_content_delta_empty_upload_is_all_deleted():
    assert content_delta({"r1": "h1"}, {}) == {
        "unchanged": 0, "changed": 0, "deleted": 1, "inserted": 0,
    }


# ----- column delta ---------------------------------------------------------

def test_column_delta_reports_added_and_removed():
    delta = column_delta({"a", "b", "c"}, {"b", "c", "d"})
    assert delta == {"added": ["d"], "removed": ["a"]}


def test_column_delta_ignores_internal_columns():
    delta = column_delta(
        {"_record_id", "_created_time", "note"}, {"Record ID", "note"}
    )
    assert delta == {"added": ["Record ID"], "removed": []}


# ----- wording --------------------------------------------------------------

def test_describe_content_delta_uses_deleting_inserting_updating():
    line = describe_content_delta(
        {"unchanged": 1142, "changed": 32, "deleted": 22, "inserted": 8}
    )
    assert "Deleting 22 rows" in line
    assert "inserting 8 new rows" in line
    assert "updating 32" in line
    assert "1,142 rows unchanged" in line


def test_describe_content_delta_says_so_when_nothing_changes():
    line = describe_content_delta(
        {"unchanged": 0, "changed": 0, "deleted": 0, "inserted": 0}
    )
    assert "No changes" in line


def test_column_change_note_is_none_when_columns_match():
    assert column_change_note({"added": [], "removed": []}) is None


def test_column_change_note_explains_the_all_updated_case():
    note = column_change_note({"added": ["New Q"], "removed": ["Old Q"]})
    assert "adds 1 column" in note and "drops 1" in note
    assert "every row counts as updated" in note


def test_column_change_note_truncates_long_lists():
    note = column_change_note(
        {"added": ["a", "b", "c", "d", "e"], "removed": []}
    )
    assert "adds 5 column(s)" in note
    assert "…" in note


def test_replace_notice_states_the_current_size():
    assert "1,204 rows" in replace_notice("housing_2026", 1204)


def test_replace_notice_handles_an_unreadable_count():
    assert "could not be read" in replace_notice("tips", None)


def test_confirm_label_carries_the_counts():
    label = confirm_label(
        "housing_2026", {"unchanged": 1142, "changed": 32, "deleted": 22, "inserted": 8}
    )
    assert "delete 22" in label
    assert "insert 8" in label
    assert "update 32" in label
    assert "housing_2026" in label


def test_confirm_label_without_a_delta_invents_no_numbers():
    label = confirm_label("tips", None)
    assert "tips" in label
    assert not any(ch.isdigit() for ch in label)


def test_wording_stays_neutral():
    texts = [
        replace_notice("t", 10),
        describe_content_delta(
            {"unchanged": 1, "changed": 1, "deleted": 1, "inserted": 1}
        ),
        confirm_label("t", {"unchanged": 1, "changed": 1, "deleted": 1, "inserted": 1}),
    ]
    for text in texts:
        assert "no backup" not in text.lower()
        assert "gone" not in text.lower()


# ----- synced tables are not upload targets -------------------------------

def test_uploadable_tables_excludes_synced_tables():
    tables = ["housing_2026", "extAbC123DeF456Gh", "schools_survey"]
    assert uploadable_tables(tables) == ["housing_2026", "schools_survey"]


def test_uploadable_tables_keeps_order():
    tables = ["zebra", "alpha", "extAbC123DeF456Gh"]
    assert uploadable_tables(tables) == ["zebra", "alpha"]


def test_a_synced_table_name_is_rejected_outright():
    df = _frame(**{"Record ID": ["r1"], "note": ["a"]})
    problems = preflight_warnings(df, "Record ID", "extAbC123DeF456Gh", {})

    assert [level for level, _ in problems] == ["error"]
    assert "sync" in problems[0][1]


def test_the_ownership_check_runs_before_everything_else():
    df = _frame(a=["1"])   # missing the id column too
    problems = preflight_warnings(df, "Record ID", "extAbC123DeF456Gh", {})
    assert len(problems) == 1
    assert "kept up to date by a sync" in problems[0][1]


def test_a_normal_table_name_is_not_rejected():
    df = _frame(**{"Record ID": ["r1"], "note": ["a"]})
    assert preflight_warnings(df, "Record ID", "housing_2026", {}) == []


def test_collisions_with_a_synced_table_are_still_reported(monkeypatch):
    monkeypatch.setattr(import_plan_module.config, "DOCUMENT_ID_SCHEME", "legacy")
    df = _frame(**{"Record ID": ["recABC", "r2"], "note": ["a", "b"]})
    problems = preflight_warnings(
        df, "Record ID", "housing_2026", {"extAbC123DeF456Gh": {"recABC"}}
    )
    assert any(
        level == "error" and "extAbC123DeF456Gh" in msg for level, msg in problems
    )


# ----- record id charset -----------------------------------------------------
#
# An id reaches a Vespa document id, a `parent_ref` inside a document-select
# string literal, and a sync_state key. Those call sites escape it, but the
# upload gate refuses the characters outright so safety doesn't depend on every
# future one remembering to.

def test_quote_in_an_id_is_an_error():
    df = _frame(**{"Record ID": ['r1" or chunk.chunk_id!="'], "note": ["a"]})
    problems = preflight_warnings(df, "Record ID", "tips", {})
    assert any(
        level == "error" and "record ID" in msg for level, msg in problems
    )


def test_backslash_in_an_id_is_an_error():
    df = _frame(**{"Record ID": ["rec\\"], "note": ["a"]})
    problems = preflight_warnings(df, "Record ID", "tips", {})
    assert _levels(problems) == {"error"}


def test_newline_in_an_id_is_an_error():
    df = _frame(**{"Record ID": ["rec\nmore"], "note": ["a"]})
    problems = preflight_warnings(df, "Record ID", "tips", {})
    assert _levels(problems) == {"error"}


def test_the_message_names_an_example_and_the_characters():
    df = _frame(**{"Record ID": ['bad"id'], "note": ["a"]})
    problems = preflight_warnings(df, "Record ID", "tips", {})
    msg = next(m for level, m in problems if level == "error")
    assert 'bad"id' in msg and '"' in msg


def test_ordinary_record_ids_are_unaffected():
    df = _frame(**{
        "Record ID": ["recABC123", "up-0001", "a_b.c", "rec:1+2@x"],
        "note": ["a", "b", "c", "d"],
    })
    assert preflight_warnings(df, "Record ID", "tips", {}) == []


# ----- tables the project's own scripts own ----------------------------------
#
# These are absent from list_tables, which is what made them dangerous: the
# page could not see them, so an upload named after one was planned as a fresh
# create — no replace notice, no confirmation — while replace_table dropped the
# real table underneath.

@pytest.mark.parametrize(
    "name", ["dataset_catalog", "enrichment", "field_inventory", "question_templates"]
)
def test_a_script_owned_table_is_refused_as_a_destination(name):
    df = _frame(**{"Record ID": ["r1"], "note": ["a"]})
    problems = preflight_warnings(df, "Record ID", name, {})
    assert any(level == "error" for level, _ in problems)


def test_a_script_owned_destination_blocks_the_plan():
    db = _FakeDB(tables=["tips"], hidden=["enrichment"])
    p = _plan(table="enrichment", db=db)

    assert p.blocked is True
    assert p.may_proceed(confirmed=True) is False


def test_the_collision_scan_covers_tables_the_page_cannot_show(monkeypatch):
    monkeypatch.setattr(import_plan_module.config, "DOCUMENT_ID_SCHEME", "legacy")
    db = _FakeDB(
        tables=["tips"],
        hidden=["question_templates"],
        ids={"question_templates": {"r1"}},
    )
    p = _plan(_df(ids=("r1",)), table="uploads", db=db)

    assert p.blocked is True
    assert any("question_templates" in msg for _, msg in p.problems)


def test_an_ordinary_upload_is_unaffected():
    db = _FakeDB(tables=["tips"], hidden=["enrichment"])
    p = _plan(_df(ids=("u1", "u2")), table="uploads", db=db)
    assert p.blocked is False
