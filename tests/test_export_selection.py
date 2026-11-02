"""The include-in-export choice: one owner, one key, one lifetime."""

import pytest

from re_source.export_selection import EXPORT_PREFIX, ExportSelection


class _Result:
    def __init__(self, rec=None, doc_id="doc-1", source_file="tips_2024"):
        self.fields = {"_record_id": rec} if rec else {}
        self.doc_id = doc_id
        self.source_file = source_file


# ----- identity --------------------------------------------------------------

def test_the_key_is_the_record_not_the_position():
    r = _Result(rec="recA")
    sel = ExportSelection({})
    assert sel.key(r) == f"{EXPORT_PREFIX}tips_2024::recA"


def test_the_key_is_stable_wherever_the_row_sits():
    r = _Result(rec="recA")
    sel = ExportSelection({})
    first = sel.key(r)
    # Same row, seen again after a filter change put it seventh.
    assert sel.key(_Result(rec="recA")) == first


def test_a_row_without_a_record_id_keys_on_its_doc_id():
    sel = ExportSelection({})
    assert sel.key(_Result(doc_id="tips__recX")) == f"{EXPORT_PREFIX}tips_2024::tips__recX"


def test_two_keyless_rows_get_different_keys():
    sel = ExportSelection({})
    assert sel.key(_Result(doc_id="a")) != sel.key(_Result(doc_id="b"))


def test_rows_from_different_bases_do_not_collide():
    sel = ExportSelection({})
    a = _Result(rec="rec1", source_file="app_one")
    b = _Result(rec="rec1", source_file="app_two")
    assert sel.key(a) != sel.key(b)


def test_a_row_with_no_identity_has_no_key():
    sel = ExportSelection({})
    assert sel.key(_Result(doc_id="")) is None
    assert sel.includes(_Result(doc_id="")) is False


# ----- the default -----------------------------------------------------------

def test_everything_is_included_until_something_is_dropped():
    rows = [_Result(rec="recA"), _Result(rec="recB")]
    assert ExportSelection({}).rows(rows) == rows


def test_rows_keeps_the_order_it_was_given():
    rows = [_Result(rec=f"rec{i}") for i in "abcde"]
    assert ExportSelection({}).rows(rows) == rows


# ----- toggling --------------------------------------------------------------

def test_toggle_drops_the_row_from_the_export():
    a, b = _Result(rec="recA"), _Result(rec="recB")
    sel = ExportSelection({})
    sel.toggle(a)
    assert sel.rows([a, b]) == [b]


def test_toggle_twice_puts_it_back():
    a = _Result(rec="recA")
    sel = ExportSelection({})
    sel.toggle(a)
    sel.toggle(a)
    assert sel.rows([a]) == [a]


def test_set_is_explicit_where_toggle_is_relative():
    a = _Result(rec="recA")
    sel = ExportSelection({})
    sel.set(a, False)
    sel.set(a, False)
    assert sel.rows([a]) == []


# ----- the bug this module exists to fix -------------------------------------

def test_the_card_view_and_the_columns_view_write_the_same_key():
    state = {}
    row = _Result(rec="recA")

    ExportSelection(state).toggle(row)          # dropped on the columns page

    assert ExportSelection(state).includes(row) is False   # ...on the card page too


def test_a_selection_survives_a_row_leaving_the_filtered_set():
    state = {}
    row = _Result(rec="recA")
    sel = ExportSelection(state)

    sel.toggle(row)
    # A filter hides the row: nothing is rendered for it, nothing reads it.
    assert sel.rows([]) == []
    # The filter is reset and it comes back — still dropped.
    assert sel.includes(row) is False


def test_a_selection_survives_load_more_extending_the_list():
    state = {}
    first_page = [_Result(rec="recA")]
    sel = ExportSelection(state)
    sel.toggle(first_page[0])

    extended = first_page + [_Result(rec="recB")]
    assert [r.fields["_record_id"] for r in sel.rows(extended)] == ["recB"]


# ----- clearing --------------------------------------------------------------

def test_clear_forgets_every_choice():
    state = {}
    sel = ExportSelection(state)
    sel.toggle(_Result(rec="recA"))
    sel.toggle(_Result(rec="recB"))

    sel.clear()

    assert state == {}


def test_clear_leaves_everything_that_is_not_a_choice_alone():
    state = {"vespa_results": ["r"], "q::Content": "vouchers"}
    sel = ExportSelection(state)
    sel.toggle(_Result(rec="recA"))

    sel.clear()

    assert state == {"vespa_results": ["r"], "q::Content": "vouchers"}
