from re_source.chunker import ParentRow
from re_source.db_loader import Document
from re_source.sync_state import SyncState
from re_source.vespa_client import FeedResult

import re_source.embed_vespa as ev


class _Client:
    def __init__(self):
        self.fed = {}

    def delete_by_selection(self, _schema, selection):
        if "bad" in selection:
            raise RuntimeError("delete failed")

    def delete(self, _schema, _doc_id):
        pass

    def feed_batch(self, schema, docs):
        self.fed[schema] = list(docs)
        return FeedResult(len(self.fed[schema]), 0, [])


def _doc(doc_id):
    return Document(
        id=doc_id,
        source_file="tips",
        fields={"_record_id": doc_id, "response": "changed"},
        content_hash=f"new-{doc_id}",
        text="response: changed",
    )


def test_failed_stale_delete_is_not_fed_or_checkpointed(monkeypatch):
    docs = [_doc("good"), _doc("bad")]
    state = SyncState(indexed={"good": "old-good", "bad": "old-bad"})
    saved = []
    monkeypatch.setattr(ev, "get_all_documents", lambda: docs)
    monkeypatch.setattr(ev, "load_sync_state", lambda _path: state)
    monkeypatch.setattr(ev, "save_sync_state", lambda value, _path: saved.append(value))
    monkeypatch.setattr(
        ev,
        "chunk_documents",
        lambda values: [
            ([], ParentRow(id=d.id, source_file=d.source_file, fields=d.fields))
            for d in values
        ],
    )

    client = _Client()
    stats = ev.feed_incremental(client, checkpoint_every=10)

    assert stats["delete_failures"] == 1
    assert [doc_id for doc_id, _fields in client.fed["parent"]] == ["good"]
    assert state.indexed == {"good": "new-good", "bad": "old-bad"}
    assert saved


# ----- author profiles survive a re-feed ---------------------------------------


def _feed_with_profiles(monkeypatch, docs, state, profiles):
    monkeypatch.setattr(ev, "get_all_documents", lambda: docs)
    monkeypatch.setattr(ev, "load_sync_state", lambda _path: state)
    monkeypatch.setattr(ev, "save_sync_state", lambda value, _path: None)
    monkeypatch.setattr(ev, "_profiles_for", lambda values: dict(profiles))
    monkeypatch.setattr(
        ev,
        "chunk_documents",
        lambda values: [
            ([], ParentRow(id=d.id, source_file=d.source_file, fields=d.fields))
            for d in values
        ],
    )
    client = _Client()
    ev.feed_incremental(client, checkpoint_every=10)
    return client


def test_a_changed_row_keeps_its_author_profile_chunk(monkeypatch):
    # The row is in state, so its chunks are cascade-deleted before re-feed.
    state = SyncState(indexed={"good": "old-good"})

    client = _feed_with_profiles(monkeypatch, [_doc("good")], state, {"good": "Works at Target."})

    profile_chunks = [fields for _id, fields in client.fed["chunk"]
                      if fields["column_key"] == "author_profile"]
    assert len(profile_chunks) == 1
    assert profile_chunks[0]["parent_id"] == "good"
    assert "Works at Target." in profile_chunks[0]["text"]


def test_a_row_without_a_current_profile_gets_no_profile_chunk(monkeypatch):
    client = _feed_with_profiles(monkeypatch, [_doc("good")], SyncState(), {})

    assert all(fields["column_key"] != "author_profile" for _id, fields in client.fed["chunk"])


def test_the_profile_stays_out_of_the_parents_keyword_text(monkeypatch):
    client = _feed_with_profiles(monkeypatch, [_doc("good")], SyncState(), {"good": "Works at Target."})

    (_id, parent), = client.fed["parent"]
    assert "author_profile" not in parent["full_row"]
    assert "Works at Target." not in parent["keyword_text"]
