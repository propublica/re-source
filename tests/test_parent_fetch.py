"""Chunk hits become parent-level results: batching, degradation, and shape."""

from re_source.vespa_chunks import (
    normalize_full_row,
    aggregate_chunks_to_parents,
    fetch_parent_rows,
)


class _RecordingClient:
    """A Vespa stand-in that records its queries."""

    def __init__(self, *, batch_rows=None, get_rows=None, query_raises=False):
        self.batch_rows = batch_rows or {}      # parent_id -> full_row value
        self.get_rows = get_rows or {}          # parent_id -> full_row value
        self.query_raises = query_raises
        self.queries = []
        self.gets = []

    def query(self, **kwargs):
        self.queries.append(kwargs)
        if self.query_raises:
            raise RuntimeError("Vespa said no")
        children = [
            {"fields": {"parent_id": pid, "full_row": row}}
            for pid, row in self.batch_rows.items()
        ]
        return {"root": {"children": children}}

    def get(self, schema, doc_id):
        self.gets.append((schema, doc_id))
        if doc_id not in self.get_rows:
            return None
        return {"full_row": self.get_rows[doc_id]}


def _ids_in(query_kwargs):
    """How many parent ids one batched query asked for."""
    return query_kwargs["yql"].count('"')// 2


# ----- batching --------------------------------------------------------------

def test_a_small_set_is_one_round_trip():
    client = _RecordingClient(batch_rows={"p1": {"A": "1"}, "p2": {"A": "2"}})
    rows = fetch_parent_rows(client, ["p1", "p2"])

    assert len(client.queries) == 1
    assert rows == {"p1": {"A": "1"}, "p2": {"A": "2"}}


def test_ids_are_batched_two_hundred_at_a_time():
    ids = [f"p{i}" for i in range(201)]
    client = _RecordingClient(batch_rows={i: {"A": "x"} for i in ids})

    fetch_parent_rows(client, ids)

    assert len(client.queries) == 2
    assert [_ids_in(q) for q in client.queries] == [200, 1]


def test_no_ids_is_no_round_trip():
    client = _RecordingClient()
    assert fetch_parent_rows(client, []) == {}
    assert client.queries == []


# ----- the per-document fallback --------------------------------------------

def test_rows_found_in_the_batch_skip_the_per_document_get():
    client = _RecordingClient(batch_rows={"p1": {"A": "1"}})

    fetch_parent_rows(client, ["p1"])

    assert client.gets == []


def test_ids_missing_from_the_batch_fall_back_to_get():
    client = _RecordingClient(batch_rows={"p1": {"A": "1"}},
                              get_rows={"p2": {"A": "2"}})

    rows = fetch_parent_rows(client, ["p1", "p2"])

    assert client.gets == [("parent", "p2")]
    assert rows == {"p1": {"A": "1"}, "p2": {"A": "2"}}


def test_a_failing_batch_degrades_to_get_rather_than_raising():
    client = _RecordingClient(query_raises=True, get_rows={"p1": {"A": "1"}})

    rows = fetch_parent_rows(client, ["p1"])

    assert rows == {"p1": {"A": "1"}}


def test_a_parent_absent_everywhere_yields_an_empty_row_not_a_crash():
    client = _RecordingClient()
    assert fetch_parent_rows(client, ["gone"]) == {"gone": {}}


# ----- the two renderings of full_row ---------------------------------------

def test_full_row_accepts_the_object_form():
    assert normalize_full_row({"Body": "x"}) == {"Body": "x"}


def test_full_row_accepts_the_array_of_pairs_form():
    assert normalize_full_row([{"key": "Body", "value": "x"}]) == {"Body": "x"}


def test_an_unrecognised_full_row_shape_is_empty_not_a_crash():
    assert normalize_full_row(None) == {}
    assert normalize_full_row("nope") == {}
    assert normalize_full_row([{"no_key": 1}]) == {}


def test_the_array_form_survives_the_round_trip():
    client = _RecordingClient(batch_rows={"p1": [{"key": "Body", "value": "x"}]})
    assert fetch_parent_rows(client, ["p1"]) == {"p1": {"Body": "x"}}


# ----- aggregation -----------------------------------------------------------

def _chunk(parent, score, *, column="Body", text="some text", chunk_id=None):
    return {
        "chunk_id": chunk_id or f"{parent}_{column.lower()}",
        "parent_id": parent,
        "column_name": column,
        "column_key": column.lower(),
        "text": text,
        "source_file": "tips_2024",
        "colbert_score": score,
    }


def test_chunks_group_under_their_parent():
    client = _RecordingClient(batch_rows={"p1": {"A": "1"}})
    (parent,) = aggregate_chunks_to_parents(
        client, [_chunk("p1", 0.9, column="Body"), _chunk("p1", 0.4, column="State")]
    )

    assert parent.parent_id == "p1"
    assert [c["column_name"] for c in parent.matching_chunks] == ["Body", "State"]


def test_a_parent_scores_as_its_best_chunk():
    client = _RecordingClient(batch_rows={"p1": {}})
    (parent,) = aggregate_chunks_to_parents(
        client, [_chunk("p1", 0.4), _chunk("p1", 0.9, column="State")]
    )
    assert parent.score == 0.9


def test_parents_come_back_best_first():
    client = _RecordingClient(batch_rows={"p1": {}, "p2": {}})
    parents = aggregate_chunks_to_parents(
        client, [_chunk("p1", 0.4), _chunk("p2", 0.9)]
    )
    assert [p.parent_id for p in parents] == ["p2", "p1"]


def test_max_results_caps_the_parent_list_at_two_hundred():
    chunks = [_chunk(f"p{i}", i / 1000) for i in range(201)]
    client = _RecordingClient(batch_rows={f"p{i}": {} for i in range(201)})

    parents = aggregate_chunks_to_parents(client, chunks)

    assert len(parents) == 200
    # The cap keeps the best, not the first seen.
    assert parents[0].parent_id == "p200"


def test_the_cap_is_applied_before_the_parents_are_fetched():
    chunks = [_chunk(f"p{i}", i / 1000) for i in range(201)]
    client = _RecordingClient(batch_rows={f"p{i}": {} for i in range(201)})

    aggregate_chunks_to_parents(client, chunks)

    assert sum(_ids_in(q) for q in client.queries) == 200


def test_no_chunks_is_no_parents_and_no_round_trip():
    client = _RecordingClient()
    assert aggregate_chunks_to_parents(client, []) == []
    assert client.queries == []
