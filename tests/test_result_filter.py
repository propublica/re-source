"""Narrowing a result set — the predicate that decides what gets disclosed."""

from datetime import date

from re_source.datasets import dataset_label
from re_source.result_filter import Narrowing, apply, offer


class _Result:
    def __init__(self, rec="recA", source_file="tips_2024", created=""):
        self.doc_id = f"doc-{rec}"
        self.source_file = source_file
        self.fields = {"_record_id": rec, "_created_time": created}
        self.payload = {}
        self.text = ""
        self.score = 0.5


def _on(day):
    return f"{day}T12:00:00.000Z"


# ----- what there is to narrow by --------------------------------------------

def test_the_bases_on_offer_come_from_the_results():
    rows = [_Result(source_file="app_one"), _Result(source_file="app_two")]
    assert len(offer(rows).datasets) == 2


def test_a_base_appears_once_however_many_rows_it_has():
    rows = [_Result(rec=f"r{i}", source_file="app_one") for i in range(5)]
    assert len(offer(rows).datasets) == 1


def test_the_bases_on_offer_are_ordered():
    rows = [_Result(source_file="zzz"), _Result(source_file="aaa")]
    assert list(offer(rows).datasets) == sorted(offer(rows).datasets)


def test_nothing_to_offer_for_no_results():
    assert offer([]).datasets == ()


# ----- narrowing by nothing --------------------------------------------------

def test_an_empty_narrowing_keeps_everything():
    rows = [_Result(rec="a"), _Result(rec="b")]
    assert apply(rows, Narrowing()).results == rows


def test_narrowing_does_not_mutate_what_it_was_given():
    rows = [_Result(rec="a")]
    apply(rows, Narrowing(datasets=()))
    assert len(rows) == 1


# ----- narrowing by dataset -----------------------------------------------------

def test_removing_a_dataset_removes_its_rows():
    keep, drop = _Result(rec="a", source_file="app_one"), _Result(rec="b", source_file="app_two")
    out = apply([keep, drop], Narrowing(datasets=(dataset_label("app_one"),)))
    assert out.results == [keep]


def test_removing_every_base_shows_nothing():
    rows = [_Result(rec="a"), _Result(rec="b")]
    assert apply(rows, Narrowing(datasets=())).results == []


# ----- narrowing by date -----------------------------------------------------

def test_a_row_inside_the_range_is_kept():
    r = _Result(created=_on("2024-03-15"))
    out = apply([r], Narrowing(dates=(date(2024, 3, 1), date(2024, 3, 31))))
    assert out.results == [r]


def test_a_row_outside_the_range_is_dropped():
    r = _Result(created=_on("2024-06-15"))
    out = apply([r], Narrowing(dates=(date(2024, 3, 1), date(2024, 3, 31))))
    assert out.results == []


def test_the_range_includes_both_of_its_ends():
    first = _Result(rec="a", created=_on("2024-03-01"))
    last = _Result(rec="b", created=_on("2024-03-31"))
    out = apply([first, last], Narrowing(dates=(date(2024, 3, 1), date(2024, 3, 31))))
    assert out.results == [first, last]


# ----- the exclusion that used to live only in a help string -----------------

def test_an_undated_row_leaves_the_export_when_a_date_bound_is_set():
    dated = _Result(rec="a", created=_on("2024-03-15"))
    undated = _Result(rec="b", created="")

    out = apply([dated, undated], Narrowing(dates=(date(2024, 3, 1), date(2024, 3, 31))))

    assert out.results == [dated]
    assert out.undated == 1


def test_a_row_with_an_unparseable_date_is_excluded_the_same_way():
    bad = _Result(rec="b", created="not a date")
    out = apply([bad], Narrowing(dates=(date(2024, 3, 1), date(2024, 3, 31))))
    assert out.results == []
    assert out.undated == 1


def test_undated_rows_are_kept_when_no_date_bound_is_set():
    undated = _Result(rec="b", created="")
    out = apply([undated], Narrowing())
    assert out.results == [undated]
    assert out.undated == 0


def test_nothing_is_reported_as_undated_when_nothing_was_excluded():
    dated = _Result(created=_on("2024-03-15"))
    out = apply([dated], Narrowing(dates=(date(2024, 3, 1), date(2024, 3, 31))))
    assert out.undated == 0


# ----- the two together ------------------------------------------------------

def test_dataset_and_date_narrow_together():
    keep = _Result(rec="a", source_file="app_one", created=_on("2024-03-15"))
    wrong_dataset = _Result(rec="b", source_file="app_two", created=_on("2024-03-15"))
    wrong_date = _Result(rec="c", source_file="app_one", created=_on("2024-09-15"))

    out = apply(
        [keep, wrong_dataset, wrong_date],
        Narrowing(datasets=(dataset_label("app_one"),),
                  dates=(date(2024, 3, 1), date(2024, 3, 31))),
    )
    assert out.results == [keep]


def test_a_row_dropped_by_dataset_is_not_counted_as_undated():
    out = apply(
        [_Result(rec="b", source_file="app_two", created="")],
        Narrowing(datasets=(dataset_label("app_one"),),
                  dates=(date(2024, 3, 1), date(2024, 3, 31))),
    )
    assert out.undated == 0


# ----- a dataset that arrives with Load more ------------------------------------
#
# "Load more" validates a deeper slice of the same pool, so it can surface a
# dataset the first page never reached. The filter defaults to every dataset present
# — meaning "everything" — but the widget stores what was selected when that
# meant fewer datasets. Without reconciling, a newly-arrived dataset arrives already
# unticked and its rows leave the view, the export and the audit record with
# nobody having chosen that.

from re_source.result_filter import widen


def test_a_base_seen_for_the_first_time_joins_the_selection():
    assert widen(selected=("A",), previously_offered=("A",),
                 now_offered=("A", "B")) == ("A", "B")


def test_a_base_the_reporter_deselected_stays_deselected():
    assert widen(selected=("A",), previously_offered=("A", "B"),
                 now_offered=("A", "B")) == ("A",)


def test_a_deselected_base_stays_out_even_as_others_arrive():
    assert widen(selected=("A",), previously_offered=("A", "B"),
                 now_offered=("A", "B", "C")) == ("A", "C")


def test_nothing_changes_when_no_new_base_arrives():
    assert widen(selected=("A", "B"), previously_offered=("A", "B"),
                 now_offered=("A", "B")) == ("A", "B")


def test_the_first_render_takes_everything_on_offer():
    assert widen(selected=(), previously_offered=(),
                 now_offered=("A", "B")) == ("A", "B")


def test_an_empty_selection_still_admits_a_new_base():
    assert widen(selected=(), previously_offered=("A",),
                 now_offered=("A", "B")) == ("B",)


def test_the_order_follows_the_menu():
    assert widen(selected=("C",), previously_offered=("C",),
                 now_offered=("A", "B", "C")) == ("A", "B", "C")


def test_a_base_that_disappears_is_dropped_from_the_selection():
    assert widen(selected=("A", "B"), previously_offered=("A", "B"),
                 now_offered=("A",)) == ("A",)


def test_the_reconciled_selection_keeps_the_new_rows(tmp_path):
    first_page = [_Result(rec="a", source_file="app_one")]
    after_more = first_page + [_Result(rec="b", source_file="app_two")]

    offered_then = offer(first_page).datasets
    offered_now = offer(after_more).datasets
    reconciled = widen(offered_then, offered_then, offered_now)

    kept = apply(after_more, Narrowing(datasets=reconciled)).results
    assert [r.fields["_record_id"] for r in kept] == ["a", "b"]


# ----- reading the date widget ------------------------------------------------
#
# A range-mode st.date_input returns 0, 1 or 2 dates depending on how far
# through the selection the reporter is. Only 2 is a range. The caller used to
# collapse everything else to None inline — right about the filtering, silent
# about the middle case: a start date picked and no end showed an unfiltered
# list while the help text promised undated rows were excluded.

from datetime import date as _date

from re_source.result_filter import DateChoice, date_choice


def test_two_dates_are_a_range():
    got = date_choice((_date(2026, 1, 1), _date(2026, 3, 1)))
    assert got.dates == (_date(2026, 1, 1), _date(2026, 3, 1))
    assert got.incomplete is False


def test_one_date_is_incomplete_not_a_range():
    got = date_choice((_date(2026, 1, 1),))
    assert got.dates is None
    assert got.incomplete is True


def test_no_dates_is_neither():
    got = date_choice(())
    assert got.dates is None
    assert got.incomplete is False


def test_a_lone_date_is_not_a_range():
    """st.date_input in single-date mode returns a bare date, not a tuple."""
    got = date_choice(_date(2026, 1, 1))
    assert got.dates is None
    assert got.incomplete is False


def test_none_entries_are_ignored():
    """Streamlit can hand back (date, None) mid-selection."""
    got = date_choice((_date(2026, 1, 1), None))
    assert got.dates is None
    assert got.incomplete is True


def test_an_incomplete_choice_filters_nothing():
    """The bug, stated as a property: a half-picked range must not narrow, and
    must not silently pretend to."""
    rows = [_Result(rec="recA", created=_on("2020-01-01")), _Result(rec="recB")]
    chosen = date_choice((_date(2026, 1, 1),))

    narrowed = apply(rows, Narrowing(datasets=None, dates=chosen.dates))
    assert len(narrowed.results) == 2
    assert narrowed.undated == 0
    assert chosen.incomplete is True          # ...and the screen can say so


# ----- reconciling the stored selection ---------------------------------------
#
# widen had nine tests proving a function nobody could get wrong. The
# read-widen-write-back that wraps it was nine lines inside a render function
# with no test surface at all — and the ordering constraint it depends on (write
# before the widget draws) existed only as a comment.

from re_source.result_filter import DATASETS_KEY, DATES_KEY, OFFERED_KEY, reconcile


def test_a_first_run_offers_every_base_and_stores_nothing_selected():
    state = {}
    rows = [_Result(source_file="app_one"), _Result(source_file="app_two")]

    on_offer = reconcile(state, rows)

    assert set(on_offer.datasets) == {"app_one", "app_two"}
    assert DATASETS_KEY not in state              # the widget's default takes over
    assert state[OFFERED_KEY] == list(on_offer.datasets)


def test_a_newly_arrived_base_is_added_to_the_stored_selection():
    """Load more can surface a dataset the first page never reached. It must not
    arrive already unticked."""
    state = {DATASETS_KEY: ["app_one"], OFFERED_KEY: ["app_one"]}
    rows = [_Result(source_file="app_one"), _Result(source_file="app_two")]

    reconcile(state, rows)

    assert state[DATASETS_KEY] == ["app_one", "app_two"]


def test_a_deselected_base_stays_deselected():
    """It was offered before, so it is not first-seen."""
    state = {DATASETS_KEY: ["app_one"], OFFERED_KEY: ["app_one", "app_two"]}
    rows = [_Result(source_file="app_one"), _Result(source_file="app_two")]

    reconcile(state, rows)

    assert state[DATASETS_KEY] == ["app_one"]


def test_what_was_offered_is_recorded_for_next_time():
    state = {DATASETS_KEY: ["app_one"], OFFERED_KEY: ["app_one"]}
    rows = [_Result(source_file="app_one"), _Result(source_file="app_two")]

    reconcile(state, rows)

    assert state[OFFERED_KEY] == ["app_one", "app_two"]


def test_the_offered_record_is_updated_even_with_nothing_stored():
    """Otherwise the first Load more would see an empty 'previously offered'
    and treat every dataset as first-seen."""
    state = {}
    reconcile(state, [_Result(source_file="app_one")])
    assert state[OFFERED_KEY] == ["app_one"]


def test_reconcile_writes_the_key_the_widget_reads():
    """The ordering constraint only pays off if both halves name the same key.
    Pinned against the constant the page passes as the widget's key."""
    state = {DATASETS_KEY: ["app_one"], OFFERED_KEY: []}
    reconcile(state, [_Result(source_file="app_one")])
    assert DATASETS_KEY in state


def test_an_empty_result_set_offers_nothing_and_keeps_the_selection_empty():
    state = {DATASETS_KEY: ["app_one"], OFFERED_KEY: ["app_one"]}
    on_offer = reconcile(state, [])

    assert on_offer.datasets == ()
    assert state[DATASETS_KEY] == []


def test_the_reset_list_covers_every_key_this_module_owns():
    """search_session.FILTER_KEYS decides what "Reset filters" clears. It names
    these strings a second time, for a different job, and consolidating would
    mean that generic state module importing this domain one. So instead: if
    the two ever disagree, a filter survives a reset that should not."""
    from re_source.search_session import FILTER_KEYS

    assert {DATASETS_KEY, DATES_KEY, OFFERED_KEY} <= set(FILTER_KEYS)

    from re_source.result_filter import EXCLUDE_KEY_PREFIX
    from re_source.search_session import FILTER_KEY_PREFIXES

    assert EXCLUDE_KEY_PREFIX in FILTER_KEY_PREFIXES


# ----- group exclusions (a deployment's FILTER_POLICY_MODULE) ----------------

def _with_filter_policy(monkeypatch, exclusions):
    import sys
    import types

    from re_source import config, result_filter

    module = types.ModuleType("test_filter_policy")
    module.group_exclusions = lambda: exclusions
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(config, "FILTER_POLICY_MODULE", module.__name__)
    result_filter._load_group_exclusions.cache_clear()


def test_no_filter_policy_offers_no_group_exclusions(monkeypatch):
    from re_source import config, result_filter

    monkeypatch.setattr(config, "FILTER_POLICY_MODULE", "")
    assert result_filter.group_exclusions() == []


def test_a_filter_policy_offers_its_group_exclusions(monkeypatch):
    from re_source import result_filter

    _with_filter_policy(monkeypatch, [("Exclude callouts", "wsp1")])

    (exclusion,) = result_filter.group_exclusions()
    assert exclusion.label == "Exclude callouts"
    assert exclusion.group == "wsp1"
    assert exclusion.key == f"{result_filter.EXCLUDE_KEY_PREFIX}wsp1"


def test_a_filter_policy_must_define_group_exclusions(monkeypatch):
    import sys
    import types

    import pytest

    from re_source import config, result_filter

    module = types.ModuleType("test_bad_filter_policy")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(config, "FILTER_POLICY_MODULE", module.__name__)
    result_filter._load_group_exclusions.cache_clear()

    with pytest.raises(ValueError, match="group_exclusions"):
        result_filter.group_exclusions()


def test_without_groups_drops_only_the_excluded_groups_rows(monkeypatch):
    from re_source import result_filter

    groups = {"app_callout": "wsp1", "app_other": "wsp2"}
    monkeypatch.setattr(result_filter, "dataset_group", groups.get)
    rows = [_Result("r1", "app_callout"), _Result("r2", "app_other"), _Result("r3", "upload")]

    assert [r.fields["_record_id"] for r in result_filter.without_groups(rows, ["wsp1"])] == ["r2", "r3"]
    assert result_filter.without_groups(rows, []) == rows
