"""feed_source_file's opt-in sync_state recording."""

import pytest

import re_source.embed_vespa as ev
from re_source.db_loader import Document
from re_source.sync_state import SyncState, load_sync_state, save_sync_state


class _FakeResult:
    def __init__(self, successes, failures=0):
        self.successes = successes
        self.failures = failures
        self.failure_examples = []


class _StubClient:
    """Records what was fed and reports a configurable failure count."""

    def __init__(self, failures=0):
        self.failures = failures
        self.fed = {}
        self.deleted = []

    def feed_batch(self, schema, docs):
        items = list(docs)
        self.fed[schema] = items
        return _FakeResult(len(items) - self.failures, self.failures)

    def delete_by_selection(self, schema, selection):
        self.deleted.append((schema, selection))

    def delete(self, schema, doc_id):
        self.deleted.append((schema, doc_id))


def _doc(doc_id, table="tips", note="mold in the unit"):
    return Document(
        id=doc_id,
        source_file=table,
        fields={"_record_id": doc_id, "note": note},
        content_hash=f"hash-{doc_id}",
        text=f"note: {note}",
    )


@pytest.fixture
def state_path(tmp_path, monkeypatch):
    """Point the sync-state read/write at a throwaway file."""
    path = tmp_path / "vespa_sync_state.json"
    monkeypatch.setattr(ev, "VESPA_SYNC_STATE_PATH", path)
    return path


@pytest.fixture
def docs(monkeypatch):
    """Two docs in `tips`, one in another table that must be ignored."""
    all_docs = [_doc("r1"), _doc("r2"), _doc("h1", table="housing")]
    monkeypatch.setattr(ev, "get_all_documents", lambda: all_docs)
    return all_docs


def test_default_does_not_record_state(state_path, docs):
    client = _StubClient()
    stats = ev.feed_source_file("tips", client)

    assert stats["state_recorded"] is False
    assert not state_path.exists()
    assert load_sync_state(state_path).indexed == {}


def test_parent_feed_carries_every_field_in_plain_keyword_text(state_path, docs):
    client = _StubClient()
    ev.feed_source_file("tips", client)

    _, parent = client.fed["parent"][0]
    assert "_record_id: r1" in parent["keyword_text"]
    assert "note: mold in the unit" in parent["keyword_text"]


def test_source_feed_removes_rows_no_longer_in_duckdb(state_path, docs):
    client = _StubClient()

    ev.feed_source_file("tips", client)

    assert client.deleted[:2] == [
        ("chunk", 'chunk.source_file=="tips"'),
        ("parent", 'parent.source_file=="tips"'),
    ]


def test_source_name_is_escaped_in_delete_selection(state_path, monkeypatch):
    name = 'tips" or parent.source_file!="'
    monkeypatch.setattr(ev, "get_all_documents", lambda: [_doc("r1", table=name)])
    client = _StubClient()

    ev.feed_source_file(name, client)

    assert client.deleted[0] == (
        "chunk",
        'chunk.source_file=="tips\\" or parent.source_file!=\\""',
    )


def test_record_state_records_only_the_fed_table(state_path, docs):
    client = _StubClient()
    stats = ev.feed_source_file("tips", client, record_state=True)

    assert stats["state_recorded"] is True
    indexed = load_sync_state(state_path).indexed
    # h1 belongs to `housing` and was never fed.
    assert indexed == {"r1": "hash-r1", "r2": "hash-r2"}


def test_a_failed_feed_does_not_record_state(state_path, docs):
    client = _StubClient(failures=1)
    stats = ev.feed_source_file("tips", client, record_state=True)

    assert stats["state_recorded"] is False
    assert stats["chunk_failures"] == 1
    assert load_sync_state(state_path).indexed == {}


def test_failed_replacement_leaves_its_rows_pending_retry(state_path, docs):
    save_sync_state(
        SyncState(
            indexed={
                "r1": "hash-r1",
                "r2": "hash-r2",
                "h1": "hash-h1",
            }
        ),
        state_path,
    )

    stats = ev.feed_source_file("tips", _StubClient(failures=1), record_state=True)

    assert stats["state_recorded"] is False
    assert load_sync_state(state_path).indexed == {"h1": "hash-h1"}


def test_unknown_source_file_raises(state_path, docs):
    with pytest.raises(RuntimeError, match="No documents found"):
        ev.feed_source_file("not_a_table", _StubClient())
