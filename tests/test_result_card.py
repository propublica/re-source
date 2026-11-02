"""What one result shows, derived once for both views."""

from datetime import date

import pytest

from re_source.export_selection import ExportSelection
from re_source.result_card import (
    NOTHING_TO_SAY,
    Match,
    card_for,
    chunk_answer,
    dom_ident,
    submitted_date,
)


class _Result:
    def __init__(self, rec="recA", *, chunks=None, payload=None, text="",
                 source_file="tips_2024", created="", doc_id="doc-1"):
        self.doc_id = doc_id
        self.source_file = source_file
        self.text = text
        self.score = 0.5
        self.fields = {"_created_time": created}
        if rec:
            self.fields["_record_id"] = rec
        self.payload = dict(payload or {})
        if chunks is not None:
            self.payload["matching_chunks"] = chunks


def _card(result=None, index=1, state=None):
    return card_for(index, result or _Result(), ExportSelection(state or {}))


# ----- the answer inside a chunk ---------------------------------------------

def test_a_json_array_value_is_unwrapped():
    assert chunk_answer('["University of Georgia"]', "") == "University of Georgia"


def test_a_column_label_is_stripped_before_unwrapping():
    assert chunk_answer('School: ["University of Georgia"]', "School") == "University of Georgia"


def test_a_question_shaped_label_is_stripped_too():
    assert chunk_answer("Where do you work? Acme Health", "Where do you work?") == "Acme Health"


def test_a_plain_value_passes_through():
    assert chunk_answer("Role: Teacher", "Role") == "Teacher"


def test_a_value_that_merely_starts_like_its_column_is_left_alone():
    assert chunk_answer("Rolex watches", "Role") == "Rolex watches"


def test_no_column_means_nothing_to_strip():
    assert chunk_answer("just a value", "") == "just a value"


# ----- identity --------------------------------------------------------------

def test_the_identity_is_the_record_not_the_position():
    assert dom_ident(_Result(rec="recA"), 1) == dom_ident(_Result(rec="recA"), 7)


def test_the_identity_is_dom_safe():
    ident = dom_ident(_Result(rec="rec/A B:C"), 1)
    assert all(ch.isalnum() or ch in "-_" for ch in ident), ident
    assert not ident.startswith("-") and not ident.endswith("-")


def test_a_keyless_row_falls_back_to_its_doc_id():
    assert "doc-9" in dom_ident(_Result(rec=None, doc_id="doc-9"), 1)


def test_two_keyless_rows_get_different_identities():
    assert dom_ident(_Result(rec=None, doc_id="a"), 1) != dom_ident(_Result(rec=None, doc_id="b"), 1)


def test_a_row_with_no_identity_at_all_is_still_drawn():
    a = dom_ident(_Result(rec=None, doc_id=""), 1)
    b = dom_ident(_Result(rec=None, doc_id=""), 2)

    assert a and b
    assert a != b          # two of them must not collide on one container key


def test_a_row_with_no_identity_is_never_in_the_export():
    sel = ExportSelection({})
    assert sel.includes(_Result(rec=None, doc_id="")) is False


def test_an_unidentifiable_row_cannot_be_ticked_back_in():
    sel = ExportSelection({})
    row = _Result(rec=None, doc_id="")
    sel.set(row, True)
    assert sel.includes(row) is False


def test_the_card_identity_and_the_export_key_agree():
    r = _Result(rec="recA")
    assert dom_ident(r, 1).replace("-", "") in ExportSelection.key(r).replace("::", "").replace("-", "")


# ----- when it was submitted -------------------------------------------------

def test_the_submitted_date_is_parsed():
    assert submitted_date(_Result(created="2024-03-15T12:00:00.000Z")) == date(2024, 3, 15)


def test_a_missing_date_is_unknown_not_old():
    assert submitted_date(_Result(created="")) is None


def test_an_unparseable_date_is_unknown_not_old():
    assert submitted_date(_Result(created="not a date")) is None


# ----- the matches -----------------------------------------------------------

def test_each_chunk_becomes_a_question_and_an_answer():
    r = _Result(chunks=[
        {"column_name": "What happened?", "text": "What happened? mold everywhere"},
        {"column_name": "State", "text": "State: CA"},
    ])
    assert _card(r).matches == (
        Match("What happened?", "mold everywhere"),
        Match("State", "CA"),
    )


def test_a_chunk_with_no_column_name_still_renders():
    r = _Result(chunks=[{"text": "I worked at Acme"}])
    assert _card(r).matches == (Match("", "I worked at Acme"),)


def test_no_chunks_means_no_matches_but_a_fallback():
    card = _card(_Result(chunks=[], text='["something"]'))
    assert card.matches == ()
    assert card.fallback == "something"


def test_the_matched_fields_line_names_up_to_five():
    r = _Result(chunks=[{"column_name": f"Q{i}", "text": "x"} for i in range(8)])
    assert _card(r).matched_fields == "Q0, Q1, Q2, Q3, Q4"


def test_the_matched_fields_line_says_so_when_there_are_none():
    assert _card(_Result(chunks=[])).matched_fields == NOTHING_TO_SAY


def test_unnamed_chunks_do_not_leave_empty_gaps_in_the_fields_line():
    r = _Result(chunks=[{"column_name": "Job", "text": "x"}, {"text": "y"}])
    assert _card(r).matched_fields == "Job"


# ----- what the strategy said ------------------------------------------------

def test_the_reason_comes_through():
    assert _card(_Result(payload={"llm_reason": "tenant"})).why == "tenant"


def test_a_strategy_that_ran_no_llm_says_so_rather_than_showing_nothing():
    assert _card(_Result()).why == NOTHING_TO_SAY


def test_the_evidence_label_names_what_the_strategy_produced():
    assert _card(_Result()).evidence_label == "Top matching chunks"
    assert _card(_Result(payload={"evidence": []})).evidence_label == "Matching evidence"


def test_cited_evidence_is_what_the_card_shows():
    card = _card(_Result(payload={
        "evidence": [{"field": "Your job", "quote": "3rd grade teacher"}],
        "matching_chunks": [],
    }))
    assert card.matches == (Match(question="Your job", answer="3rd grade teacher"),)
    assert card.matched_fields == "Your job"


def test_highlights_come_through_as_given():
    assert _card(_Result(payload={"highlights": ["mold"]})).highlights == ("mold",)


def test_a_strategy_with_no_highlights_yields_none_rather_than_none_the_value():
    assert _card(_Result()).highlights == ()


# ----- the export state ------------------------------------------------------

def test_a_card_knows_whether_it_is_in_the_export():
    r = _Result(rec="recA")
    state = {}
    assert _card(r, state=state).included is True

    ExportSelection(state).toggle(r)
    assert _card(r, state=state).included is False


# ----- the rest --------------------------------------------------------------

def test_the_index_is_the_one_it_was_given():
    assert _card(index=7).index == 7


def test_the_base_is_humanised():
    assert _card(_Result(source_file="tips_2024")).dataset


def test_the_full_row_is_carried_for_the_modal():
    r = _Result(rec="recA")
    assert _card(r).fields["_record_id"] == "recA"
