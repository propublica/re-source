"""Model-cited evidence is checked against the row before it is shown."""

from re_source.llm_utils import (
    GENERATED_FIELDS,
    corroborated_evidence,
    normalize_for_quote_match,
)


ROW = {
    "Your job": "I am a 3rd grade teacher in Dayton.",
    "What happened?": "The district cut our supply budget in half.",
    "_record_id": "rec1",
    "author_profile": "A teacher, mid-career.",
}


# ----- genuine citations survive ---------------------------------------------

def test_a_verbatim_quote_from_a_real_field_is_kept():
    out = corroborated_evidence(ROW, [{"field": "Your job", "quote": "3rd grade teacher"}])
    assert out == [{"field": "Your job", "quote": "3rd grade teacher"}]


def test_several_citations_keep_their_order():
    out = corroborated_evidence(ROW, [
        {"field": "What happened?", "quote": "supply budget"},
        {"field": "Your job", "quote": "teacher in Dayton"},
    ])
    assert [e["field"] for e in out] == ["What happened?", "Your job"]


def test_reflowed_whitespace_and_case_still_match():
    out = corroborated_evidence(
        ROW, [{"field": "your job", "quote": "3RD   GRADE\n teacher"}]
    )
    assert len(out) == 1


def test_the_whole_field_may_be_quoted():
    out = corroborated_evidence(
        ROW, [{"field": "Your job", "quote": "I am a 3rd grade teacher in Dayton."}]
    )
    assert len(out) == 1


# ----- fabrications do not -----------------------------------------------------

def test_a_quote_that_is_not_in_the_field_is_dropped():
    out = corroborated_evidence(ROW, [
        {"field": "Your job", "quote": "I am a nurse at St. Jude and my cell is 555-0134"},
    ])
    assert out == []


def test_a_quote_from_the_wrong_field_is_dropped():
    out = corroborated_evidence(
        ROW, [{"field": "Your job", "quote": "supply budget"}]
    )
    assert out == []


def test_an_invented_field_is_dropped():
    out = corroborated_evidence(
        ROW, [{"field": "Home address", "quote": "12 Elm St"}]
    )
    assert out == []


def test_an_underscore_field_is_never_citable():
    out = corroborated_evidence(ROW, [{"field": "_record_id", "quote": "rec1"}])
    assert out == []


def test_an_empty_field_or_quote_is_dropped():
    out = corroborated_evidence(ROW, [
        {"field": "Your job", "quote": ""},
        {"field": "", "quote": "teacher"},
    ])
    assert out == []


# ----- shapes that must not crash the strategy-------------------------------

def test_no_evidence_at_all_is_empty_not_an_error():
    assert corroborated_evidence(ROW, None) == []
    assert corroborated_evidence(ROW, []) == []


def test_a_non_dict_citation_is_skipped():
    out = corroborated_evidence(
        ROW, ["garbage", {"field": "Your job", "quote": "teacher"}]
    )
    assert len(out) == 1


def test_a_row_with_no_fields_corroborates_nothing():
    assert corroborated_evidence({}, [{"field": "Your job", "quote": "teacher"}]) == []
    assert corroborated_evidence(None, [{"field": "a", "quote": "b"}]) == []


def test_a_non_string_field_value_does_not_crash():
    out = corroborated_evidence(
        {"Rooms": 3, "When": None}, [{"field": "Rooms", "quote": "3"}]
    )
    assert out == [{"field": "Rooms", "quote": "3"}]


def test_normalize_collapses_runs_of_whitespace():
    assert normalize_for_quote_match("  A   B \n C ") == "a b c"
    assert normalize_for_quote_match(None) == ""


# ----- text this system wrote is not the tipster's words ---------------------
#
# The Author Profile strategy hides author_profile from the validator and then
# renders the cited evidence as the submitter's own quoted prose. A citation of
# author_profile therefore quotes the model's summary back as the tipster's
# words, with the real row text overwritten — the exact leak the strategy says it
# prevents. The underscore rule missed it: the field has no underscore.

def test_the_generated_profile_cannot_corroborate_itself():
    out = corroborated_evidence(
        ROW, [{"field": "author_profile", "quote": "A teacher, mid-career."}]
    )
    assert out == []


def test_a_generated_field_is_not_citable_in_any_case_spelling():
    out = corroborated_evidence(
        ROW, [{"field": "Author_Profile", "quote": "A teacher"}]
    )
    assert out == []


def test_real_fields_still_corroborate_alongside_a_rejected_one():
    """One bad citation must not discard the genuine ones beside it."""
    out = corroborated_evidence(ROW, [
        {"field": "author_profile", "quote": "A teacher, mid-career."},
        {"field": "Your job", "quote": "3rd grade teacher"},
    ])
    assert out == [{"field": "Your job", "quote": "3rd grade teacher"}]


def test_the_verify_text_hides_exactly_what_corroboration_refuses():
    """The two rules are one rule. They were separate literals, which is how
    the citation side came to miss author_profile while the prompt side hid
    it."""
    from re_source.vespa_strategies import _author_profile_verify_text
    from re_source.search_types import SearchResult

    result = SearchResult(
        doc_id="d1", text="", source_file="tips", fields=dict(ROW), score=1.0,
    )
    (shown,) = _author_profile_verify_text([result])

    for name in GENERATED_FIELDS:
        assert name not in shown.lower()
        assert corroborated_evidence(ROW, [{"field": name, "quote": "x"}]) == []
