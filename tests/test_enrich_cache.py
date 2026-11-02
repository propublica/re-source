"""The profile cache is keyed on row content, not on the Vespa index format."""

import hashlib
import json

import duckdb

from re_source import enrich
from re_source.db_loader import document_from_row


def _doc(**fields):
    return document_from_row({"_record_id": "recA", "Story": "I drive for a warehouse", **fields}, "tips")


def _pre_index_format_hash(row: dict) -> str:
    # compute_content_hash as it was before INDEX_FORMAT_VERSION was folded in.
    return hashlib.sha256(json.dumps(row, sort_keys=True, default=str).encode()).hexdigest()[:16]


def test_row_hash_is_the_pre_index_format_content_hash():
    doc = _doc()

    assert enrich.row_hash(doc) == _pre_index_format_hash(doc.fields)


def test_a_profile_cached_before_the_index_format_bump_is_current():
    doc = _doc()

    assert not enrich.needs_enrichment(doc, _pre_index_format_hash(doc.fields))


def test_a_profile_cached_under_the_index_format_hash_is_current():
    doc = _doc()

    assert not enrich.needs_enrichment(doc, doc.content_hash)


def test_a_changed_row_or_missing_profile_needs_enrichment():
    old = _doc()
    new = _doc(Story="I drive for a different warehouse now")

    assert enrich.needs_enrichment(new, enrich.row_hash(old))
    assert enrich.needs_enrichment(new, None)


def test_enrich_rows_skips_profiles_cached_before_the_bump(monkeypatch, tmp_path):
    db = tmp_path / "tips.duckdb"
    cached, changed, fresh = _doc(), _doc(_record_id="recB"), _doc(_record_id="recC")
    conn = duckdb.connect(str(db))
    enrich._ensure_table(conn)
    enrich._upsert_enrichment(conn, cached.id, _pre_index_format_hash(cached.fields), "kept")
    enrich._upsert_enrichment(conn, changed.id, "stale-hash", "old")
    conn.close()

    calls = []
    monkeypatch.setattr(enrich, "DUCKDB_PATH", db)
    monkeypatch.setattr(enrich, "generate_author_profile", lambda text: calls.append(text) or "new")

    stats = enrich.enrich_rows([cached, changed, fresh], max_workers=1)

    assert stats["skipped_unchanged"] == 1
    assert len(calls) == 2
    conn = duckdb.connect(str(db), read_only=True)
    stored = dict(conn.execute("SELECT doc_id, source_hash FROM enrichment").fetchall())
    conn.close()
    # New entries are written under the content-only hash.
    assert stored[fresh.id] == enrich.row_hash(fresh)


def test_pending_enrichment_lists_only_rows_without_a_current_profile(monkeypatch, tmp_path):
    db = tmp_path / "tips.duckdb"
    current, stale, missing = _doc(), _doc(_record_id="recB"), _doc(_record_id="recC")
    conn = duckdb.connect(str(db))
    enrich._ensure_table(conn)
    enrich._upsert_enrichment(conn, current.id, enrich.row_hash(current), "kept")
    enrich._upsert_enrichment(conn, stale.id, "stale-hash", "old")
    conn.close()
    monkeypatch.setattr(enrich, "DUCKDB_PATH", db)

    assert [d.id for d in enrich.pending_enrichment([current, stale, missing])] == [stale.id, missing.id]
