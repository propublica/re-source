"""One convention for what a chunk's text carries, read and written."""

import pytest

from re_source.field_value import (
    is_blank_value,
    label_for,
    with_label,
    without_label,
)


# ----- the label -------------------------------------------------------------

def test_a_question_keeps_its_own_mark():
    assert label_for("Are you a parent?") == "Are you a parent?"


def test_a_heading_takes_a_colon():
    assert label_for("Shelter experience") == "Shelter experience:"


def test_no_column_is_no_label():
    assert label_for("") == ""
    assert label_for(None) == ""


# ----- writing ---------------------------------------------------------------

def test_a_question_joins_with_a_space():
    assert with_label("Are you a parent?", "Yes") == "Are you a parent? Yes"


def test_a_heading_joins_with_a_colon():
    assert with_label("Shelter experience", "I lived in one") == "Shelter experience: I lived in one"


def test_the_label_rides_along_so_a_short_answer_has_context():
    assert "parent" in with_label("Are you a parent?", "Yes")


def test_an_answer_with_no_column_is_just_the_answer():
    assert with_label("", "bare value") == "bare value"


def test_surrounding_whitespace_does_not_survive():
    assert with_label("  Role  ", "  Teacher  ") == "Role: Teacher"


# ----- reading back ----------------------------------------------------------

def test_a_question_label_comes_off():
    assert without_label("Are you a parent? Yes", "Are you a parent?") == "Yes"


def test_a_heading_label_comes_off():
    assert without_label("Shelter experience: I lived in one", "Shelter experience") == "I lived in one"


def test_text_that_does_not_carry_this_label_is_left_alone():
    assert without_label("About the person who submitted this", "Role") == "About the person who submitted this"


def test_a_value_that_merely_starts_like_its_column_is_left_alone():
    assert without_label("Rolex watches", "Role") == "Rolex watches"


def test_no_column_means_nothing_to_strip():
    assert without_label("just a value", "") == "just a value"


# ----- the round trip --------------------------------------------------------

@pytest.mark.parametrize("column,value", [
    ("Are you a parent?", "Yes"),
    ("Shelter experience", "I lived in one for six months"),
    ("What happened?", "mold everywhere"),
    ("Role", "Teacher"),
])
def test_writing_then_reading_gives_back_the_answer(column, value):
    assert without_label(with_label(column, value), column) == value


def test_an_answer_that_starts_with_a_question_mark_survives():
    assert without_label(with_label("Role", "?? no idea"), "Role") == "?? no idea"


def test_an_answer_that_starts_with_a_colon_survives():
    assert without_label(with_label("Role", ": see above"), "Role") == ": see above"


def test_an_answer_containing_the_label_again_survives():
    assert without_label(with_label("Role", "Role: Teacher"), "Role") == "Role: Teacher"


# ----- what counts as no answer ----------------------------------------------

def test_an_empty_value_is_blank():
    assert is_blank_value("") is True
    assert is_blank_value(None) is True
    assert is_blank_value("   ") is True


def test_null_like_tokens_are_blank():
    for token in ("N/A", "na", "none", "null", "nil", "nan", "-", "no answer"):
        assert is_blank_value(token) is True, token


def test_a_genuine_short_answer_is_not_blank():
    assert is_blank_value("No") is False
    assert is_blank_value("Yes") is False


def test_json_array_wrappers_are_unwrapped_first():
    assert is_blank_value('["No"]') is False
    assert is_blank_value('[null]') is True
    assert is_blank_value('[]') is True
    assert is_blank_value('["N/A"]') is True


def test_an_empty_object_is_blank():
    assert is_blank_value("{}") is True


def test_something_that_only_looks_like_json_is_judged_as_text():
    assert is_blank_value("[not json after all") is False


# ----- the four parties agree ------------------------------------------------

def test_the_chunker_writes_through_this_convention():
    from re_source.chunker import format_chunk_text

    assert format_chunk_text("Role", "Teacher") == with_label("Role", "Teacher")


def test_the_enrichment_chunk_is_readable_back():
    from re_source.enrich import PROFILE_LABEL

    text = with_label(PROFILE_LABEL, "a nurse in Utah")
    assert without_label(text, PROFILE_LABEL) == "a nurse in Utah"


def test_the_screen_reads_a_chunk_back_through_this_convention():
    from re_source.result_card import chunk_answer

    assert chunk_answer(with_label("Role", "Teacher"), "Role") == "Teacher"


def test_the_search_path_reads_a_chunk_back_the_same_way():
    from re_source.lane_union import is_contentless
    from re_source.search_types import SearchResult

    def _result(column, value):
        return SearchResult(
            doc_id="d", text="", source_file="t", fields={}, score=0.0,
            payload={"matching_chunks": [
                {"column_name": column, "text": with_label(column, value)}
            ]},
        )

    assert is_contentless(_result("Are you a parent?", "[null]")) is True
    assert is_contentless(_result("Are you a parent?", "No")) is False
    # The answer that the old strip ate.
    assert is_contentless(_result("Role", "?? no idea")) is False


def test_the_search_path_does_not_import_the_indexing_module():
    import inspect

    import re_source.vespa_strategies as vs

    assert "from .chunker import" not in inspect.getsource(vs)
