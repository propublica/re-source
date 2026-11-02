"""Replacing a table is atomic, and the column layout is one rule."""

import duckdb
import pytest

from re_source.duckdb_table import LEADING_COLUMNS, ordered_columns, replace_table


@pytest.fixture
def conn(tmp_path):
    c = duckdb.connect(str(tmp_path / "t.duckdb"))
    yield c
    c.close()


def _rows(conn, sql):
    return conn.execute(sql).fetchall()


# ----- the column layout -----------------------------------------------------

def test_the_id_and_time_columns_lead():
    assert ordered_columns(["zebra", "apple"])[:2] == list(LEADING_COLUMNS)


def test_everything_else_is_alphabetical():
    assert ordered_columns(["zebra", "apple", "Mango"]) == [
        "_record_id", "_created_time", "Mango", "apple", "zebra"
    ]


def test_the_leading_columns_are_not_duplicated_when_supplied():
    assert ordered_columns(["_record_id", "note", "_created_time"]) == [
        "_record_id", "_created_time", "note"
    ]


def test_the_leading_columns_are_added_when_absent():
    assert ordered_columns(["note"]) == ["_record_id", "_created_time", "note"]


# ----- the swap --------------------------------------------------------------

def test_replace_writes_the_rows(conn):
    replace_table(conn, "tips", ["_record_id", "note"], [["r1", "a"], ["r2", "b"]])
    assert _rows(conn, "SELECT _record_id, note FROM tips ORDER BY _record_id") == [
        ("r1", "a"), ("r2", "b")
    ]


def test_replace_replaces_rather_than_appends(conn):
    replace_table(conn, "tips", ["_record_id", "note"], [["r1", "a"]])
    replace_table(conn, "tips", ["_record_id", "note"], [["r9", "z"]])
    assert _rows(conn, "SELECT _record_id FROM tips") == [("r9",)]


def test_an_empty_row_set_leaves_an_empty_table_not_a_missing_one(conn):
    replace_table(conn, "tips", ["_record_id", "note"], [])
    assert _rows(conn, "SELECT count(*) FROM tips") == [(0,)]


def test_a_failed_swap_leaves_the_previous_table_intact(conn):
    replace_table(conn, "tips", ["_record_id", "note"], [["r1", "a"], ["r2", "b"]])

    with pytest.raises(duckdb.Error):
        replace_table(conn, "tips", ["_record_id", "note", "note"],
                      [["r3", "x", "y"]])

    assert _rows(conn, "SELECT count(*) FROM tips") == [(2,)]
    assert _rows(conn, "SELECT _record_id FROM tips ORDER BY _record_id") == [
        ("r1",), ("r2",)
    ]


def test_a_failed_insert_also_rolls_the_table_back(conn):
    replace_table(conn, "tips", ["_record_id", "note"], [["r1", "a"]])

    with pytest.raises(duckdb.Error):
        replace_table(conn, "tips", ["_record_id", "note"], [["r2"]])

    assert _rows(conn, "SELECT _record_id FROM tips") == [("r1",)]


def test_the_connection_stays_usable_after_a_failure(conn):
    with pytest.raises(duckdb.Error):
        replace_table(conn, "tips", ["a", "a"], [["x", "y"]])

    replace_table(conn, "other", ["_record_id", "note"], [["r1", "a"]])
    assert _rows(conn, "SELECT _record_id FROM other") == [("r1",)]


# ----- identifiers -----------------------------------------------------------

def test_a_quote_in_a_column_name_cannot_break_out(conn):
    replace_table(conn, "tips", ["_record_id", 'we"ird'], [["r1", "v"]])
    assert _rows(conn, 'SELECT "we""ird" FROM tips') == [("v",)]


def test_a_quote_in_a_table_name_cannot_break_out(conn):
    replace_table(conn, 'ta"ble', ["_record_id"], [["r1"]])
    assert _rows(conn, 'SELECT _record_id FROM "ta""ble"') == [("r1",)]
