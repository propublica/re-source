"""Counting rows for the progress readout goes through quote_ident."""

import importlib.util
from pathlib import Path

import duckdb
import pytest

_spec = importlib.util.spec_from_file_location(
    "feed_progress",
    Path(__file__).resolve().parent.parent / "scripts" / "feed_progress.py",
)
fp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fp)


@pytest.fixture
def db(tmp_path, monkeypatch):
    path = tmp_path / "scratch.duckdb"
    conn = duckdb.connect(str(path))
    try:
        conn.execute('CREATE TABLE tips ("_record_id" VARCHAR)')
        conn.executemany("INSERT INTO tips VALUES (?)", [["r1"], ["r2"]])
        conn.execute('CREATE TABLE enrichment ("doc_id" VARCHAR)')
        conn.execute("INSERT INTO enrichment VALUES ('r1')")
        conn.execute('CREATE TABLE field_inventory ("source_file" VARCHAR)')
        conn.execute("INSERT INTO field_inventory VALUES ('tips')")
    finally:
        conn.close()
    monkeypatch.setattr(fp, "DUCKDB_PATH", path)
    return path


def test_counts_only_source_tables(db):
    assert fp.total_rows() == 2


def test_a_table_name_with_a_quote_is_counted_not_executed(db):
    weird = 'x" UNION SELECT count(*) FROM enrichment --'
    conn = duckdb.connect(str(db))
    try:
        from re_source.db_loader import quote_ident

        conn.execute(f'CREATE TABLE {quote_ident(weird)} ("_record_id" VARCHAR)')
        conn.execute(f"INSERT INTO {quote_ident(weird)} VALUES ('r1')")
    finally:
        conn.close()

    # 2 from tips + 1 from the oddly-named table. An injected UNION would
    # change this number, which is the tell.
    assert fp.total_rows() == 3
