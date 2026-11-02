"""re_source/importer.py writes spreadsheet headers into DuckDB DDL."""

import csv
from pathlib import Path

import duckdb
import pytest

import re_source.importer as import_file_mod


@pytest.fixture
def db(tmp_path, monkeypatch):
    """Point the importer at a throwaway DuckDB."""
    path = tmp_path / "scratch.duckdb"
    monkeypatch.setattr(import_file_mod, "DUCKDB_PATH", path)
    return path


def _csv(tmp_path, header: str, value: str = "v1") -> Path:
    """Write a two-column CSV."""
    path = tmp_path / "in.csv"
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", header])
        writer.writerow(["r1", value])
    return path


def test_quote_in_column_header_does_not_inject_sql(tmp_path, db):
    header = 'evil" VARCHAR); DROP TABLE victim; --'
    result = import_file_mod.import_file(
        path=_csv(tmp_path, header), id_column="id", table_name="tips", sheet=None
    )
    assert result["rows"] == 1

    conn = duckdb.connect(str(db))
    try:
        # The sentinel table survives, i.e. no injected DROP ran.
        conn.execute("CREATE TABLE victim (x INTEGER)")
        cols = [r[1] for r in conn.execute("PRAGMA table_info('tips')").fetchall()]
    finally:
        conn.close()
    assert header in cols


def test_quote_in_table_name_does_not_inject_sql(tmp_path, db):
    table = 'evil" (x INTEGER); DROP TABLE victim; --'
    import_file_mod.import_file(
        path=_csv(tmp_path, "Plain"), id_column="id", table_name=table, sheet=None
    )

    conn = duckdb.connect(str(db))
    try:
        conn.execute("CREATE TABLE victim (x INTEGER)")
        tables = [r[0] for r in conn.execute("SHOW TABLES").fetchall()]
    finally:
        conn.close()
    assert table in tables


def test_ordinary_import_still_round_trips(tmp_path, db):
    import_file_mod.import_file(
        path=_csv(tmp_path, "What happened?", "mold everywhere"),
        id_column="id",
        table_name="tips",
        sheet=None,
    )
    conn = duckdb.connect(str(db))
    try:
        row = conn.execute('SELECT _record_id, "What happened?" FROM tips').fetchone()
    finally:
        conn.close()
    assert row == ("r1", "mold everywhere")
