"""The upload page's row delta must match what the pipeline really computes."""

import pandas as pd
import pytest

import re_source.db_loader as db_loader
import re_source.importer as importer


@pytest.fixture
def db(tmp_path, monkeypatch):
    """Point both modules at the same throwaway DuckDB."""
    path = tmp_path / "scratch.duckdb"
    monkeypatch.setattr(importer, "DUCKDB_PATH", path)
    monkeypatch.setattr(db_loader, "DUCKDB_PATH", path)
    return path


def _frame(rows):
    return pd.DataFrame(rows, columns=["Record ID", "What happened?", "Where?"])


BASE = [
    ["r1", "mold everywhere", "unit 4"],
    ["r2", "no heat since June", "unit 9"],
    ["r3", "eviction notice", "unit 2"],
]


def _imported(db, rows):
    """Import `rows` and return the hashes now in the table."""
    importer.import_dataframe(_frame(rows), id_column="Record ID", table_name="tips")
    return db_loader.content_hashes("tips")


def _pending(rows):
    return importer.pending_hashes(_frame(rows), "Record ID", "tips")


def test_identical_reupload_produces_identical_hashes(db):
    after_import = _imported(db, BASE)
    assert _pending(BASE) == after_import


def test_editing_one_cell_changes_exactly_one_hash(db):
    after_import = _imported(db, BASE)
    edited = [r[:] for r in BASE]
    edited[1][1] = "no heat since May"

    pending = _pending(edited)
    changed = [k for k in pending if pending[k] != after_import.get(k)]
    assert changed == [db_loader.make_document_id("tips", "r2")]


def test_a_dropped_row_shows_up_as_missing(db):
    after_import = _imported(db, BASE)
    pending = _pending(BASE[:2])

    assert set(after_import) - set(pending) == {
        db_loader.make_document_id("tips", "r3")
    }
    assert set(pending) - set(after_import) == set()


def test_a_new_row_shows_up_as_added(db):
    after_import = _imported(db, BASE)
    pending = _pending(BASE + [["r9", "flooding", "basement"]])

    assert set(pending) - set(after_import) == {
        db_loader.make_document_id("tips", "r9")
    }
    assert set(after_import) - set(pending) == set()


def test_adding_a_column_changes_every_hash(db):
    after_import = _imported(db, BASE)

    wider = _frame(BASE)
    wider["New question?"] = ""
    pending = importer.pending_hashes(wider, "Record ID", "tips")

    assert set(pending) == set(after_import)          # same ids
    assert all(pending[k] != after_import[k] for k in pending)   # every hash moved


def test_reordering_columns_changes_nothing(db):
    after_import = _imported(db, BASE)

    shuffled = _frame(BASE)[["Where?", "Record ID", "What happened?"]]
    assert importer.pending_hashes(shuffled, "Record ID", "tips") == after_import


def test_reordering_rows_changes_nothing(db):
    after_import = _imported(db, BASE)
    assert _pending(list(reversed(BASE))) == after_import


def test_textless_rows_are_absent_from_both_sides(db):
    rows = BASE + [["r4", "", ""]]
    after_import = _imported(db, rows)

    record_id = db_loader.make_document_id("tips", "r4")
    assert record_id not in after_import
    assert record_id not in _pending(rows)
