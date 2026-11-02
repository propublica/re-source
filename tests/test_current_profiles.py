"""Which saved author profiles the indexer carries onto a re-fed row."""

import duckdb

from re_source import enrich
from re_source.db_loader import document_from_row


def _doc(record_id, story="I stock shelves at Target"):
    return document_from_row({"_record_id": record_id, "Story": story}, "tips")


def _db(monkeypatch, tmp_path, rows):
    db = tmp_path / "tips.duckdb"
    conn = duckdb.connect(str(db))
    enrich._ensure_table(conn)
    for doc_id, source_hash, profile in rows:
        enrich._upsert_enrichment(conn, doc_id, source_hash, profile)
    conn.close()
    monkeypatch.setattr(enrich, "DUCKDB_PATH", db)


def test_current_non_empty_profiles_are_carried(monkeypatch, tmp_path):
    doc = _doc("recA")
    _db(monkeypatch, tmp_path, [(doc.id, enrich.row_hash(doc), "Works at Target.")])

    assert enrich.current_profiles([doc]) == {doc.id: "Works at Target."}


def test_stale_empty_and_missing_profiles_are_not(monkeypatch, tmp_path):
    stale, empty, missing = _doc("recA"), _doc("recB"), _doc("recC")
    _db(monkeypatch, tmp_path, [
        (stale.id, enrich.row_hash(_doc("recA", story="something else")), "Old summary."),
        (empty.id, enrich.row_hash(empty), ""),
    ])

    assert enrich.current_profiles([stale, empty, missing]) == {}


def test_an_unreadable_database_feeds_without_profiles(monkeypatch, tmp_path):
    bad = tmp_path / "not_a_database.duckdb"
    bad.write_text("this is not duckdb")
    monkeypatch.setattr(enrich, "DUCKDB_PATH", bad)

    assert enrich.current_profiles([_doc("recA")]) == {}
