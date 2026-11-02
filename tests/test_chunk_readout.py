"""Reading a chunk response: the hit loop that used to be copied per strategy."""

import pytest

from re_source.vespa_chunks import ChunkReadout, best_score_by_parent, read_chunk_response


class _Client:
    """Answers the parent fetch; raises if asked for anything unexpected."""

    def __init__(self, rows=None):
        self.rows = rows or {}
        self.queries = 0

    def query(self, **kwargs):
        self.queries += 1
        return {"root": {"children": [
            {"fields": {"parent_id": pid, "full_row": row}}
            for pid, row in self.rows.items()
        ]}}

    def get(self, schema, doc_id):
        return {"full_row": self.rows.get(doc_id, {})}


class _Exploding:
    """A source that must not be touched."""

    def query(self, **kwargs):
        raise AssertionError("the client was queried")

    def get(self, schema, doc_id):
        raise AssertionError("the client was asked for a document")


def _hit(i=0, *, features=None, relevance=0.0, text="answer", parent="p1", **fields):
    base = {
        "chunk_id": f"{parent}_c{i}",
        "parent_id": parent,
        "column_name": f"Q{i}",
        "column_key": f"c{i}",
        "text": text,
        "source_file": "tips_2024",
        "matchfeatures": features if features is not None else {"secondPhase": 0.9},
    }
    base.update(fields)
    return {"relevance": relevance, "fields": base}


def _response(*hits):
    return {"root": {"children": list(hits)}}


# ----- the chunk dict --------------------------------------------------------

def test_hits_become_chunk_dicts_with_the_aggregator_keys():
    out = read_chunk_response(_Client({"p1": {}}), _response(_hit()),
                              score_feature="secondPhase")

    (chunk,) = out.chunks
    assert set(chunk) == {"chunk_id", "parent_id", "column_name", "column_key",
                          "text", "source_file", "colbert_score"}


def test_null_text_becomes_an_empty_string():
    hit = _hit()
    del hit["fields"]["text"]
    out = read_chunk_response(_Client({"p1": {}}), _response(hit),
                              score_feature="secondPhase")

    assert out.chunks[0]["text"] == ""
    assert out.results[0].text == ""


def test_a_chunk_always_carries_a_chunk_id_key():
    hit = _hit()
    del hit["fields"]["chunk_id"]
    out = read_chunk_response(_Client({"p1": {}}), _response(hit),
                              score_feature="secondPhase")
    assert out.chunks[0]["chunk_id"] == ""


# ----- scoring ---------------------------------------------------------------

def test_the_named_match_feature_supplies_the_score():
    hit = _hit(features={"secondPhase": 0.9, "bm25(text)": 12.5})

    colbert = read_chunk_response(_Client({"p1": {}}), _response(hit),
                                  score_feature="secondPhase")
    lexical = read_chunk_response(_Client({"p1": {}}), _response(hit),
                                  score_feature="bm25(text)")

    assert colbert.chunks[0]["colbert_score"] == 0.9
    assert lexical.chunks[0]["colbert_score"] == 12.5


def test_relevance_is_the_fallback_when_the_feature_is_absent():
    hit = _hit(features={}, relevance=0.42)
    out = read_chunk_response(_Client({"p1": {}}), _response(hit),
                              score_feature="secondPhase")

    assert out.chunks[0]["colbert_score"] == 0.42


def test_the_score_feature_must_be_named():
    with pytest.raises(TypeError):
        read_chunk_response(_Client(), _response(), "secondPhase")


# ----- parent identity -------------------------------------------------------

def test_a_missing_parent_id_is_derived_from_the_chunk_id():
    hit = _hit(parent="tips__recA")
    hit["fields"]["chunk_id"] = "tips__recA_body"
    hit["fields"]["column_key"] = "body"
    del hit["fields"]["parent_id"]

    out = read_chunk_response(_Client({"tips__recA": {}}), _response(hit),
                              score_feature="secondPhase")

    assert out.chunks[0]["parent_id"] == "tips__recA"


# ----- the empty response ----------------------------------------------------

def test_an_empty_response_yields_an_empty_readout_without_touching_the_client():
    out = read_chunk_response(_Exploding(), {"root": {"children": []}},
                              score_feature="secondPhase")

    assert out == ChunkReadout(results=[], parents=[], chunks=[],
                               num_candidates=0, cutoff=0)


def test_a_response_with_no_children_key_is_not_an_error():
    out = read_chunk_response(_Exploding(), {}, score_feature="secondPhase")
    assert out.results == []


def test_a_null_children_list_is_not_an_error():
    out = read_chunk_response(_Exploding(), {"root": {"children": None}},
                              score_feature="secondPhase")
    assert out.results == []


# ----- the cutoff ------------------------------------------------------------

def _descending(n=20):
    """Scores with a cliff the cutoff would cut at."""
    return _response(*[
        _hit(i, features={"secondPhase": s}, parent=f"p{i}")
        for i, s in enumerate([1.0] * 10 + [0.01] * (n - 10))
    ])


def test_the_cutoff_is_not_applied_unless_requested():
    rows = {f"p{i}": {} for i in range(20)}
    out = read_chunk_response(_Client(rows), _descending(),
                              score_feature="secondPhase")

    assert len(out.chunks) == 20
    assert out.cutoff == 20


def test_the_cutoff_trims_the_tail_when_requested():
    rows = {f"p{i}": {} for i in range(20)}
    out = read_chunk_response(_Client(rows), _descending(),
                              score_feature="secondPhase", apply_cutoff=True)

    assert len(out.chunks) == 10


def test_num_candidates_counts_the_raw_hits_not_the_survivors():
    rows = {f"p{i}": {} for i in range(20)}
    out = read_chunk_response(_Client(rows), _descending(),
                              score_feature="secondPhase", apply_cutoff=True)

    assert out.num_candidates == 20
    assert out.cutoff == 10


def test_the_cutoff_is_reported_raw_not_clamped():
    out = read_chunk_response(_Client({"p1": {}}),
                              _response(_hit(features={"secondPhase": 1.0})),
                              score_feature="secondPhase", apply_cutoff=True)

    assert out.cutoff >= 1
    assert len(out.chunks) == 1


# ----- the parent cap --------------------------------------------------------

def test_max_parents_is_two_hundred_by_default():
    hits = _response(*[_hit(i, features={"secondPhase": i / 1000}, parent=f"p{i}")
                       for i in range(201)])
    out = read_chunk_response(_Client({f"p{i}": {} for i in range(201)}), hits,
                              score_feature="secondPhase")

    assert len(out.parents) == 200


def test_max_parents_can_be_narrowed():
    hits = _response(*[_hit(i, features={"secondPhase": i / 10}, parent=f"p{i}")
                       for i in range(5)])
    out = read_chunk_response(_Client({f"p{i}": {} for i in range(5)}), hits,
                              score_feature="secondPhase", max_parents=2)

    assert [p.parent_id for p in out.parents] == ["p4", "p3"]


# ----- the narrower reader ---------------------------------------------------

def test_best_score_by_parent_keeps_the_highest_per_parent():
    resp = _response(
        _hit(0, features={"firstPhase": 0.4}, parent="p1"),
        _hit(1, features={"firstPhase": 0.9}, parent="p1"),
        _hit(2, features={"firstPhase": 0.5}, parent="p2"),
    )
    assert best_score_by_parent(resp, score_feature="firstPhase") == {
        "p1": 0.9, "p2": 0.5
    }


def test_best_score_by_parent_is_empty_for_an_empty_response():
    assert best_score_by_parent({"root": {"children": []}},
                                score_feature="firstPhase") == {}


def test_best_score_by_parent_shares_the_relevance_fallback():
    resp = _response(_hit(features={}, relevance=0.33, parent="p1"))
    assert best_score_by_parent(resp, score_feature="firstPhase") == {"p1": 0.33}


def test_best_score_by_parent_derives_a_missing_parent_id():
    hit = _hit(features={"firstPhase": 0.5})
    hit["fields"]["chunk_id"] = "tips__recA_body"
    hit["fields"]["column_key"] = "body"
    del hit["fields"]["parent_id"]

    assert best_score_by_parent(_response(hit), score_feature="firstPhase") == {
        "tips__recA": 0.5
    }
