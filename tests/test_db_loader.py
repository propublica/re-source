"""Table introspection in re_source/db_loader.py."""

import duckdb
import pytest

import re_source.db_loader as db_loader


@pytest.fixture
def db(tmp_path, monkeypatch):
    """A throwaway DuckDB with two source tables and the enrichment cache."""
    path = tmp_path / "scratch.duckdb"
    conn = duckdb.connect(str(path))
    try:
        conn.execute('CREATE TABLE tips ("_record_id" VARCHAR, "note" VARCHAR)')
        conn.executemany(
            "INSERT INTO tips VALUES (?, ?)",
            [["r1", "mold"], ["r2", "no heat"], ["r3", "eviction"]],
        )
        conn.execute('CREATE TABLE housing ("_record_id" VARCHAR, "note" VARCHAR)')
        conn.executemany("INSERT INTO housing VALUES (?, ?)", [["r3", "x"], ["h9", "y"]])
        # Not tips — must never show up as a source table.
        conn.execute('CREATE TABLE enrichment ("doc_id" VARCHAR, "profile" VARCHAR)')
        conn.execute("INSERT INTO enrichment VALUES ('r1', 'a landlord')")
        # Also not tips: corpus-maintenance tables written into the same database.
        conn.execute(
            'CREATE TABLE field_inventory ("source_file" VARCHAR, "sample_values" VARCHAR)'
        )
        conn.execute("INSERT INTO field_inventory VALUES ('tips', 'a@b.com; c@d.com')")
        conn.execute(
            'CREATE TABLE question_templates ("column_key" VARCHAR, "question" VARCHAR)'
        )
        conn.execute("INSERT INTO question_templates VALUES ('note', 'What happened?')")
        conn.execute(
            'CREATE TABLE dataset_catalog ('
            'dataset_id VARCHAR PRIMARY KEY, label VARCHAR, record_url VARCHAR)'
        )
        conn.execute("INSERT INTO dataset_catalog VALUES ('tips', 'Tips', NULL)")
    finally:
        conn.close()
    monkeypatch.setattr(db_loader, "DUCKDB_PATH", path)
    return path


def test_list_tables_excludes_the_enrichment_cache(db):
    assert db_loader.list_tables() == ["housing", "tips"]


def test_list_tables_excludes_every_script_written_table(db):
    listed = db_loader.list_tables()
    assert "field_inventory" not in listed
    assert "question_templates" not in listed
    assert "dataset_catalog" not in listed


def test_the_feed_never_loads_a_non_source_table(db):
    sources = {doc.source_file for doc in db_loader.get_all_documents()}
    assert sources == {"tips", "housing"}


def test_list_tables_on_a_missing_database_is_empty_not_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(db_loader, "DUCKDB_PATH", tmp_path / "nope.duckdb")
    assert db_loader.list_tables() == []


def test_count_rows(db):
    assert db_loader.count_rows("tips") == 3
    assert db_loader.count_rows("housing") == 2


def test_count_rows_on_unknown_table_raises(db):
    with pytest.raises(duckdb.Error):
        db_loader.count_rows("does_not_exist")


def test_record_ids_returns_the_id_set(db):
    assert db_loader.record_ids("tips") == {"r1", "r2", "r3"}
    # r3 is in both tables — this is the collision the page warns about.
    assert db_loader.record_ids("tips") & db_loader.record_ids("housing") == {"r3"}


def test_record_ids_without_the_column_is_empty_not_an_error(db):
    assert db_loader.record_ids("enrichment") == set()


def test_quote_ident_escapes_embedded_quotes(db):
    weird = 'odd" name'
    conn = duckdb.connect(str(db))
    try:
        conn.execute(f'CREATE TABLE {db_loader.quote_ident(weird)} ("_record_id" VARCHAR)')
        conn.execute(f"INSERT INTO {db_loader.quote_ident(weird)} VALUES ('r1')")
    finally:
        conn.close()
    assert db_loader.count_rows(weird) == 1
    assert weird in db_loader.list_tables()


# ----- document_from_row ----------------------------------------------------

def test_document_from_row_builds_a_document():
    doc = db_loader.document_from_row(
        {"_record_id": "r1", "_created_time": "", "note": "mold"}, "tips"
    )
    assert doc is not None
    assert doc.id == db_loader.make_document_id("tips", "r1")
    assert doc.source_file == "tips"
    assert doc.text == "note: mold"
    assert len(doc.content_hash) == 16


def test_document_from_row_skips_rows_with_no_searchable_text():
    assert db_loader.document_from_row(
        {"_record_id": "r1", "_created_time": "2026-01-01", "note": ""}, "tips"
    ) is None


def test_document_from_row_prefers_underscore_over_a_column_named_id():
    """`_record_id` holds the row's real identity, so a column merely *named*
    `id` must not outrank it — it used to, which made a value a tip submitter
    typed the document id."""
    doc = db_loader.document_from_row(
        {"id": "typed-by-a-submitter", "_record_id": "recABC123", "note": "x"}, "tips"
    )
    assert doc.id == db_loader.make_document_id("tips", "recABC123")


def test_document_from_row_still_accepts_a_bare_id_column():
    """Kept as a fallback for a table written by neither importer, which may
    carry only `id`."""
    doc = db_loader.document_from_row({"id": "legacy-1", "note": "x"}, "tips")
    assert doc.id == db_loader.make_document_id("tips", "legacy-1")


def test_a_blank_record_id_falls_through_to_the_id_column():
    """An empty `_record_id` is not an identity; don't let it mask a real one."""
    doc = db_loader.document_from_row(
        {"_record_id": "", "id": "legacy-1", "note": "x"}, "tips"
    )
    assert doc.id == db_loader.make_document_id("tips", "legacy-1")


def test_document_from_row_falls_back_to_a_content_hash_id():
    row = {"note": "mold", "where": "unit 4"}
    first = db_loader.document_from_row(row, "tips")
    second = db_loader.document_from_row(dict(row), "tips")
    assert first.id == second.id
    assert first.id.startswith("ds1_")


def test_document_ids_are_scoped_to_the_dataset():
    assert db_loader.make_document_id("tips", "r1") != db_loader.make_document_id(
        "housing", "r1"
    )


def test_legacy_document_ids_remain_available(monkeypatch):
    monkeypatch.setattr(db_loader, "DOCUMENT_ID_SCHEME", "legacy")
    assert db_loader.make_document_id("tips", "r1") == "r1"


def test_content_hash_ignores_column_order():
    a = db_loader.document_from_row({"_record_id": "r1", "a": "1", "b": "2"}, "t")
    b = db_loader.document_from_row({"b": "2", "_record_id": "r1", "a": "1"}, "t")
    assert a.content_hash == b.content_hash


def test_content_hash_changes_when_a_column_is_added():
    a = db_loader.document_from_row({"_record_id": "r1", "a": "1"}, "t")
    b = db_loader.document_from_row({"_record_id": "r1", "a": "1", "extra": ""}, "t")
    assert a.content_hash != b.content_hash


def test_index_format_is_part_of_the_content_hash(monkeypatch):
    row = {"_record_id": "r1", "a": "1"}
    before = db_loader.compute_content_hash(row)
    monkeypatch.setattr(db_loader, "INDEX_FORMAT_VERSION", 999)
    assert db_loader.compute_content_hash(row) != before


def test_load_table_still_skips_textless_rows(db):
    conn = duckdb.connect(str(db))
    try:
        conn.execute('CREATE TABLE mixed ("_record_id" VARCHAR, "note" VARCHAR)')
        conn.executemany(
            "INSERT INTO mixed VALUES (?, ?)", [["r1", "mold"], ["r2", ""], ["r3", None]]
        )
        docs = list(db_loader.load_table(conn, "mixed"))
    finally:
        conn.close()
    assert [d.id for d in docs] == [db_loader.make_document_id("mixed", "r1")]


def test_column_names_in_declaration_order(db):
    assert db_loader.column_names("tips") == ["_record_id", "note"]


def test_column_names_quotes_odd_identifiers(db):
    weird = 'odd" table'
    conn = duckdb.connect(str(db))
    try:
        conn.execute(f'CREATE TABLE {db_loader.quote_ident(weird)} ("a" VARCHAR)')
    finally:
        conn.close()
    assert db_loader.column_names(weird) == ["a"]


def test_column_names_unknown_table_is_empty(db):
    assert db_loader.column_names("does_not_exist") == []


def test_record_ids_handles_odd_table_names(db):
    weird = 'odd" name'
    conn = duckdb.connect(str(db))
    try:
        conn.execute(f'CREATE TABLE {db_loader.quote_ident(weird)} ("_record_id" VARCHAR)')
        conn.execute(f"INSERT INTO {db_loader.quote_ident(weird)} VALUES ('r1')")
    finally:
        conn.close()
    assert db_loader.record_ids(weird) == {"r1"}
