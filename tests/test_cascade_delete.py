"""A parent id never widens the chunk delete that cascades from it."""

from re_source.embed_vespa import cascade_delete_parents


class FakeClient:
    """Records the selections it is asked to delete by, deleting nothing."""

    def __init__(self):
        self.selections: list[str] = []
        self.deleted: list[tuple[str, str]] = []

    def delete_by_selection(self, schema, selection):
        self.selections.append(selection)
        return 1

    def delete(self, schema, doc_id):
        self.deleted.append((schema, doc_id))


def test_ordinary_id_is_selected_exactly():
    client = FakeClient()
    assert cascade_delete_parents(client, ["tips__recABC123"]) == ["tips__recABC123"]
    assert client.selections == ['chunk.parent_ref=="id:tips:parent::tips__recABC123"']
    assert client.deleted == [("parent", "tips__recABC123")]


def test_quote_in_an_id_cannot_close_the_literal():
    client = FakeClient()
    cascade_delete_parents(client, ['r1" or chunk.chunk_id!="'])

    selection = client.selections[0]
    assert selection == (
        'chunk.parent_ref=="id:tips:parent::r1\\" or chunk.chunk_id!=\\""'
    )
    # The tell-tale of a break-out: an unescaped quote followed by another
    # clause. Every quote inside the literal must carry its backslash.
    assert ' or chunk.chunk_id!=""' not in selection
    assert selection.count('\\"') == 2


def test_backslash_is_escaped_before_the_quote():
    client = FakeClient()
    cascade_delete_parents(client, ["rec\\"])
    assert client.selections[0] == 'chunk.parent_ref=="id:tips:parent::rec\\\\"'


class Flaky(FakeClient):
    """Fails the delete for any id containing 'boom'."""

    def delete_by_selection(self, schema, selection):
        self.selections.append(selection)
        if "boom" in selection:
            raise RuntimeError("nope")
        return 1


def test_one_bad_id_does_not_stop_the_rest():
    """A failure is logged per id, not raised — the caller is mid-feed."""
    client = Flaky()
    assert cascade_delete_parents(client, ["ok1", "boom", "ok2"]) == ["ok1", "ok2"]


def test_the_failed_id_is_not_reported_as_deleted():
    """The caller prunes sync_state by this return value, and deletions are
    derived as state-minus-DuckDB — so an id reported deleted but still in the
    index can never be retried by any later run."""
    assert "boom" not in cascade_delete_parents(Flaky(), ["ok1", "boom", "ok2"])


def test_everything_failing_reports_nothing_deleted():
    assert cascade_delete_parents(Flaky(), ["boom1", "boom2"]) == []
