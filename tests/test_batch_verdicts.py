"""Aligning a batch validator's answer back onto the documents it judged."""

from re_source.llm_utils import _parse_batch_response


def _texts(n):
    return [f"doc {i}" for i in range(1, n + 1)]


def _verdict(i, relevant=True, reason="because"):
    return {"index": i, "relevant": relevant, "reason": reason}


def _json(*verdicts):
    import json
    return json.dumps({"results": list(verdicts)})


# ----- the happy path --------------------------------------------------------

def test_every_document_gets_exactly_one_verdict():
    out = _parse_batch_response(_json(_verdict(1), _verdict(2), _verdict(3)), 3)
    assert len(out) == 3
    assert [v["relevant"] for v in out] == [True, True, True]


def test_a_verdict_lands_on_the_document_its_index_names():
    out = _parse_batch_response(
        _json(_verdict(1, reason="first"), _verdict(2, reason="second")), 2
    )
    assert [v["reason"] for v in out] == ["first", "second"]


# ----- the two failure modes it was written for ------------------------------

def test_out_of_order_verdicts_are_realigned_on_their_index():
    out = _parse_batch_response(
        _json(_verdict(3, reason="third"), _verdict(1, reason="first"),
              _verdict(2, reason="second")), 3
    )
    assert [v["reason"] for v in out] == ["first", "second", "third"]


def test_a_short_answer_keeps_the_documents_it_skipped():
    out = _parse_batch_response(_json(_verdict(1, relevant=False)), 3)

    assert len(out) == 3
    assert out[0]["relevant"] is False          # the one it did judge
    assert out[1]["relevant"] is True           # the two it didn't
    assert out[2]["relevant"] is True
    assert "kept by default" in out[1]["reason"]


def test_an_answer_with_no_indices_falls_back_to_position():
    out = _parse_batch_response(
        _json({"relevant": False, "reason": "a"}, {"relevant": True, "reason": "b"}), 2
    )
    assert [v["reason"] for v in out] == ["a", "b"]


# ----- fail open, always -----------------------------------------------------

def test_an_unparseable_answer_keeps_everything():
    out = _parse_batch_response("not json at all", 3)
    assert len(out) == 3
    assert all(v["relevant"] is True for v in out)
    assert all("unparseable" in v["reason"] for v in out)


def test_an_answer_with_no_results_key_keeps_everything():
    out = _parse_batch_response('{"something_else": []}', 2)
    assert [v["relevant"] for v in out] == [True, True]


def test_results_of_the_wrong_shape_keep_everything():
    out = _parse_batch_response('{"results": "not a list"}', 2)
    assert [v["relevant"] for v in out] == [True, True]


def test_a_non_dict_entry_is_skipped_rather_than_crashing():
    out = _parse_batch_response(_json(_verdict(1), "garbage", _verdict(3)), 3)
    assert len(out) == 3
    assert out[1]["relevant"] is True           # the garbage slot is kept


def test_an_index_outside_the_batch_is_ignored():
    out = _parse_batch_response(
        _json(_verdict(1, reason="real"), _verdict(99, reason="hallucinated")), 2
    )
    assert out[0]["reason"] == "real"
    assert "kept by default" in out[1]["reason"]


def test_a_non_numeric_index_is_ignored():
    out = _parse_batch_response(
        _json({"index": "two", "relevant": False, "reason": "x"}), 2
    )
    assert [v["relevant"] for v in out] == [True, True]


def test_an_empty_batch_returns_nothing():
    assert _parse_batch_response(_json(), 0) == []


# ----- the screen's own gap rule ---------------------------------------------

def test_the_screen_can_opt_out_of_keeping_gaps():
    out = _parse_batch_response("not json", 2, default_keep=False)
    assert all(v["relevant"] is False for v in out)


def test_the_filler_reason_is_the_one_callers_recognise():
    (only,) = _parse_batch_response(_json(), 1)
    assert only["reason"] == "no verdict returned for this document — kept by default"


# ----- one verdict per document ----------------------------------------------
#
# A repeated index means the answer does not describe this batch. Letting the
# second copy win would hand one document's verdict to another — and the model
# is steerable by the documents themselves, since they share a prompt.

def test_a_repeated_index_applies_neither_verdict():
    out = _parse_batch_response(
        _json(_verdict(1, relevant=False, reason="real"),
              _verdict(1, relevant=True, reason="second opinion")), 2
    )
    assert "conflicting" in out[0]["reason"]


def test_a_repeated_index_does_not_disturb_the_other_documents():
    out = _parse_batch_response(
        _json(_verdict(1, reason="a"), _verdict(1, reason="b"),
              _verdict(2, reason="untouched")), 2
    )
    assert out[1]["reason"] == "untouched"


def test_a_conflicted_slot_follows_the_callers_default():
    out = _parse_batch_response(
        _json(_verdict(1), _verdict(1)), 1, default_keep=False
    )
    assert out[0]["relevant"] is False


def test_a_conflicted_slot_carries_no_model_supplied_keys():
    out = _parse_batch_response(
        _json({"index": 1, "relevant": True, "workplace": {"employer": "Acme"}},
              {"index": 1, "relevant": True, "workplace": {"employer": "Other"}}), 1
    )
    assert "workplace" not in out[0]
