"""re_source/importer.py's replace-the-table import is atomic."""

import duckdb
import pandas as pd
import pytest

import re_source.datasets as datasets
import re_source.importer as importer


@pytest.fixture
def synced(monkeypatch):
    """A table shape the deployment has told us a sync owns."""
    monkeypatch.setattr(datasets, "READ_ONLY_TABLE_PATTERNS", (r"^ext[A-Za-z0-9]{14}$",))


@pytest.fixture
def db(tmp_path, monkeypatch):
    """Point the importer at a throwaway DuckDB."""
    path = tmp_path / "scratch.duckdb"
    monkeypatch.setattr(importer, "DUCKDB_PATH", path)
    return path


def _rows(db, sql):
    conn = duckdb.connect(str(db))
    try:
        return conn.execute(sql).fetchall()
    finally:
        conn.close()


def test_failed_import_leaves_the_previous_table_intact(db):
    good = pd.DataFrame({"id": ["r1", "r2"], "What happened?": ["mold", "no heat"]})
    importer.import_dataframe(good, id_column="id", table_name="tips")
    assert _rows(db, "SELECT count(*) FROM tips") == [(2,)]

    # `id` renames to `_record_id`, which the frame already has -> duplicate column.
    bad = pd.DataFrame({"id": ["r3"], "_record_id": ["stale"], "note": ["x"]})
    with pytest.raises(duckdb.Error):
        importer.import_dataframe(bad, id_column="id", table_name="tips")

    # Still two rows from the good import, not zero and not one.
    assert _rows(db, "SELECT count(*) FROM tips") == [(2,)]
    assert _rows(db, "SELECT _record_id FROM tips ORDER BY _record_id") == [("r1",), ("r2",)]


def test_successful_import_replaces_rather_than_appends(db):
    first = pd.DataFrame({"id": ["r1", "r2"], "note": ["a", "b"]})
    importer.import_dataframe(first, id_column="id", table_name="tips")

    second = pd.DataFrame({"id": ["r9"], "note": ["z"]})
    stats = importer.import_dataframe(second, id_column="id", table_name="tips")

    assert stats["rows"] == 1
    assert _rows(db, "SELECT _record_id FROM tips") == [("r9",)]


def test_duplicate_ids_are_rejected(db):
    df = pd.DataFrame(
        {"id": ["r1", "r1", "", None, "r2"], "note": ["a", "b", "c", "d", "e"]}
    )
    with pytest.raises(ValueError, match="duplicate record IDs"):
        importer.import_dataframe(df, id_column="id", table_name="tips")


def test_missing_id_column_names_the_available_columns(db):
    df = pd.DataFrame({"a": ["1"], "b": ["2"]})
    with pytest.raises(ValueError, match="Record ID"):
        importer.import_dataframe(df, id_column="Record ID", table_name="tips")


def test_import_creates_the_data_directory_if_missing(tmp_path, monkeypatch):
    target = tmp_path / "brand" / "new" / "tips.duckdb"
    monkeypatch.setattr(importer, "DUCKDB_PATH", target)

    importer.import_dataframe(
        pd.DataFrame({"id": ["r1"], "note": ["a"]}),
        id_column="id",
        table_name="tips",
    )
    assert target.exists()


# ----- normalize_for_import -------------------------------------------------

def test_normalize_puts_record_id_and_created_time_first():
    df = pd.DataFrame({"zebra": ["z"], "Record ID": ["r1"], "apple": ["a"]})
    _, cols = importer.normalize_for_import(df, "Record ID")
    assert cols == ["_record_id", "_created_time", "apple", "zebra"]


def test_normalize_synthesizes_empty_created_time():
    df = pd.DataFrame({"Record ID": ["r1"], "note": ["a"]})
    rows, cols = importer.normalize_for_import(df, "Record ID")
    assert rows[0][cols.index("_created_time")] == ""


def test_normalize_keeps_an_existing_created_time():
    df = pd.DataFrame({"Record ID": ["r1"], "_created_time": ["2026-01-01"]})
    rows, cols = importer.normalize_for_import(df, "Record ID")
    assert rows[0][cols.index("_created_time")] == "2026-01-01"


def test_normalize_coerces_values_to_str_or_none():
    df = pd.DataFrame({"Record ID": ["r1"], "count": [7], "blank": [None]})
    rows, cols = importer.normalize_for_import(df, "Record ID")
    assert rows[0][cols.index("count")] == "7"
    assert rows[0][cols.index("blank")] is None


def test_normalize_does_not_mutate_the_caller_frame():
    df = pd.DataFrame({"Record ID": ["r1"], "note": ["a"]})
    before = list(df.columns)
    importer.normalize_for_import(df, "Record ID")
    assert list(df.columns) == before


# ----- destinations a file import may never replace ---------------------------
#
# The rules used to live in re_source/import_plan.py, which is imported by the upload
# page and nothing else. So the two other doors — scripts/import_file.py, and
# any future caller — went straight past them: `--table-name enrichment` dropped
# the author-profile cache, `--table-name app0Aky...` replaced a synced base.
# A rule only one of three callers consults is a convention, not a guard.

@pytest.mark.parametrize(
    "table", ["dataset_catalog", "enrichment", "field_inventory", "question_templates"]
)
def test_a_script_owned_table_is_refused(db, table):
    with pytest.raises(ValueError, match="own scripts"):
        importer.import_dataframe(
            pd.DataFrame({"id": ["r1"], "note": ["a"]}),
            id_column="id", table_name=table,
        )


def test_a_synced_table_is_refused(db, synced):
    with pytest.raises(ValueError, match="kept up to date by a sync"):
        importer.import_dataframe(
            pd.DataFrame({"id": ["r1"], "note": ["a"]}),
            id_column="id", table_name="extAbC123DeF456Gh",
        )


def test_a_case_variant_of_a_synced_table_is_refused(db, synced):
    """DuckDB resolves identifiers case-insensitively, so this names the same
    physical table."""
    with pytest.raises(ValueError, match="kept up to date by a sync"):
        importer.import_dataframe(
            pd.DataFrame({"id": ["r1"], "note": ["a"]}),
            id_column="id", table_name="ExtAbC123DeF456Gh",
        )


def test_a_refused_destination_is_not_touched(db):
    """The point of refusing before the write rather than after: the table that
    was there is still there."""
    conn = duckdb.connect(str(db))
    try:
        conn.execute('CREATE TABLE enrichment ("doc_id" VARCHAR, "profile" VARCHAR)')
        conn.execute("INSERT INTO enrichment VALUES ('r1', 'a landlord')")
    finally:
        conn.close()

    with pytest.raises(ValueError):
        importer.import_dataframe(
            pd.DataFrame({"id": ["r1"], "note": ["a"]}),
            id_column="id", table_name="enrichment",
        )

    assert _rows(db, "SELECT * FROM enrichment") == [("r1", "a landlord")]


def test_the_cli_door_is_covered_too(db, tmp_path):
    """import_file delegates to import_dataframe, which is why putting the
    guard there covers the door that never had one."""
    csv = tmp_path / "f.csv"
    csv.write_text("id,note\nr1,a\n")

    with pytest.raises(ValueError, match="own scripts"):
        importer.import_file(
            path=csv, id_column="id", table_name="enrichment", sheet=None
        )


def test_an_ordinary_destination_still_imports(db):
    importer.import_dataframe(
        pd.DataFrame({"id": ["r1"], "note": ["a"]}),
        id_column="id", table_name="uploads",
    )
    assert _rows(db, "SELECT _record_id FROM uploads") == [("r1",)]


def test_the_page_and_the_write_give_the_same_reason(db):
    """One sentence, one source. The page shows what the write would raise."""
    from re_source.import_plan import preflight_warnings

    (level, shown), = preflight_warnings(
        pd.DataFrame({"id": ["r1"], "note": ["a"]}), "id", "enrichment", {}
    )
    assert level == "error"

    with pytest.raises(ValueError) as caught:
        importer.import_dataframe(
            pd.DataFrame({"id": ["r1"], "note": ["a"]}),
            id_column="id", table_name="enrichment",
        )
    assert str(caught.value) == shown
