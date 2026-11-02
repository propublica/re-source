"""One running search: what it holds, how it pages, and what a clear leaves."""

from re_source.search_session import (
    FILTER_KEYS,
    SEARCH_KEYS,
    ActiveSearch,
    PagePlan,
    SearchSession,
)


def _started(state=None, *, pool=None, cursor=0, results=("a",)):
    state = {} if state is None else state
    s = SearchSession(state)
    s.start(results=list(results), metadata={"num_parents": 1}, elapsed=1.5,
            query="mold", strategy="Content", pool=pool, cursor=cursor)
    return s, state


# ----- what is on screen -----------------------------------------------------

def test_a_fresh_session_has_no_active_search():
    assert SearchSession({}).active() is None


def test_start_then_active_round_trips():
    s, _ = _started(pool=["c1", "c2"], cursor=1)
    active = s.active()

    assert isinstance(active, ActiveSearch)
    assert active.results == ["a"]
    assert active.elapsed == 1.5
    assert active.metadata == {"num_parents": 1}
    assert active.query == "mold"
    assert active.strategy == "Content"
    assert active.pool == ["c1", "c2"]
    assert active.cursor == 1


def test_an_empty_result_set_is_still_an_active_search():
    s = SearchSession({})
    s.start(results=[], metadata={}, elapsed=0.2, query="nothing",
            strategy="Content")
    assert s.active() is not None
    assert s.active().results == []


def test_a_missing_metadata_reads_as_empty_not_none():
    state = {"vespa_results": ["a"], "vespa_metadata": None}
    assert SearchSession(state).active().metadata == {}


# ----- paging ----------------------------------------------------------------

def test_a_full_page_is_taken_when_the_pool_is_deep():
    pool = [f"c{i}" for i in range(500)]
    s, _ = _started(pool=pool, cursor=50)

    plan = s.page_plan(50)

    assert isinstance(plan, PagePlan)
    assert (plan.cursor, plan.step, plan.remaining) == (50, 50, 450)
    assert plan.pool_slice == pool[50:100]


def test_the_last_page_is_clamped_to_what_is_left():
    pool = [f"c{i}" for i in range(120)]
    s, _ = _started(pool=pool, cursor=100)

    plan = s.page_plan(50)

    assert plan.step == 20
    assert plan.remaining == 20
    assert len(plan.pool_slice) == 20
    assert plan.pool_slice[-1] == "c119"


def test_there_is_no_page_once_the_pool_is_spent():
    pool = [f"c{i}" for i in range(50)]
    s, _ = _started(pool=pool, cursor=50)
    assert s.page_plan(50) is None


def test_there_is_no_page_for_a_strategy_that_did_not_stash_one():
    s, _ = _started(pool=None)
    assert s.page_plan(50) is None


def test_there_is_no_page_before_a_search_has_run():
    assert SearchSession({}).page_plan(50) is None


# ----- going deeper ----------------------------------------------------------

def test_extend_advances_the_cursor_by_the_step():
    pool = [f"c{i}" for i in range(120)]
    s, _ = _started(pool=pool, cursor=50)

    assert s.extend(["b"], 50) == 100
    assert s.extend(["c"], 20) == 120


def test_extend_appends_rather_than_replaces():
    s, _ = _started(results=["a"], pool=["c"] * 100, cursor=50)
    s.extend(["b", "c"], 50)
    assert s.active().results == ["a", "b", "c"]


def test_walking_the_pool_never_runs_past_it():
    pool = [f"c{i}" for i in range(120)]
    s, _ = _started(pool=pool, cursor=0)

    seen = []
    while (plan := s.page_plan(50)) is not None:
        seen.extend(plan.pool_slice)
        s.extend(plan.pool_slice, plan.step)

    assert seen == pool
    assert s.active().cursor == 120


def test_extend_tracks_the_running_total_not_the_page():
    s, _ = _started(results=["a"], pool=["c"] * 100, cursor=50)
    s.extend(["b", "c"], 50)
    assert s.active().metadata["validated_count"] == 3


def test_extend_keeps_the_rest_of_the_metadata():
    s, _ = _started(pool=["c"] * 100, cursor=50)
    s.extend(["b"], 50)
    assert s.active().metadata["num_parents"] == 1


# ----- clearing --------------------------------------------------------------

def test_clear_drops_the_whole_search():
    state = {k: "x" for k in SEARCH_KEYS}
    SearchSession(state).clear()
    assert state == {}


def test_clear_leaves_the_session_alone():
    state = {
        "vespa_results": ["r"],
        "audit_page": 3,
        "q::Content": "school vouchers",
        "search_strategy_choice": "BM25",
        "langfuse_session_id": "sess-1",
        "export::tips_2024::recA": False,
    }
    SearchSession(state).clear()

    assert state == {
        "q::Content": "school vouchers",
        "search_strategy_choice": "BM25",
        "langfuse_session_id": "sess-1",
        "export::tips_2024::recA": False,
    }


def test_clearing_a_fresh_session_is_a_no_op():
    state = {}
    SearchSession(state).clear()
    assert state == {}


def test_clear_filters_drops_only_the_filter_widgets():
    state = {"filter-datasets": ["tips_2024"], "filter-dates": ("a", "b"),
             "vespa_results": ["r"]}
    SearchSession(state).clear_filters()
    assert state == {"vespa_results": ["r"]}


def test_group_exclusions_are_reset_and_cleared_with_the_other_filters():
    state = {"filter-exclude::wsp1": True, "vespa_results": ["r"]}
    SearchSession(state).clear_filters()
    assert state == {"vespa_results": ["r"]}

    state = {"filter-exclude::wsp1": True, "vespa_results": ["r"]}
    SearchSession(state).clear()
    assert state == {}


def test_a_filter_cannot_outlive_the_results_it_narrows():
    assert set(FILTER_KEYS) <= set(SEARCH_KEYS)


# ----- session identity ------------------------------------------------------

def test_the_trace_session_id_is_minted_once():
    state = {}
    s = SearchSession(state)
    first = s.trace_session_id(lambda: "generated")
    assert s.trace_session_id(lambda: "different") == first


def test_the_remembered_strategy_round_trips():
    s = SearchSession({})
    assert s.remembered_strategy() is None
    s.remember_strategy("BM25")
    assert s.remembered_strategy() == "BM25"


# ----- what has been disclosed -----------------------------------------------

def test_a_new_search_starts_at_page_one():
    s = SearchSession({})
    assert s.begin_audit_page(new_search=True) == 1


def test_going_deeper_counts_on():
    s = SearchSession({})
    s.begin_audit_page(new_search=True)
    assert s.begin_audit_page(new_search=False) == 2
    assert s.begin_audit_page(new_search=False) == 3


def test_a_new_search_restarts_the_page_count():
    s = SearchSession({})
    s.begin_audit_page(new_search=True)
    s.begin_audit_page(new_search=False)
    assert s.begin_audit_page(new_search=True) == 1


def test_unseen_returns_everything_the_first_time():
    s = SearchSession({})
    assert s.unseen(["a::1", "a::2"]) == ["a::1", "a::2"]


def test_unseen_drops_what_was_already_recorded():
    s = SearchSession({})
    s.unseen(["a::1"])
    assert s.unseen(["a::1", "a::2"]) == ["a::2"]


def test_unseen_preserves_order():
    s = SearchSession({})
    assert s.unseen(["a::3", "a::1", "a::2"]) == ["a::3", "a::1", "a::2"]


def test_a_new_search_forgets_what_was_seen():
    s = SearchSession({})
    s.begin_audit_page(new_search=True)
    s.unseen(["a::1"])

    s.begin_audit_page(new_search=True)
    assert s.unseen(["a::1"]) == ["a::1"]


def test_there_is_no_page_when_the_strategy_has_no_page_size():
    s, _ = _started(pool=["c0", "c1"], cursor=0)
    assert s.page_plan(None) is None
