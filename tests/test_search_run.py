"""Driving a strategy: once, or a page at a time."""

import pytest

from re_source.search_runner import FirstPage, SearchRun, start_run
from re_source.vespa_strategies import VESPA_STRATEGIES


class _Paging:
    """A strategy that retrieves a pool and validates it a page at a time."""

    paginates = True
    PAGE_SIZE = 2

    def __init__(self, pool=None):
        self.pool = pool if pool is not None else [f"c{i}" for i in range(5)]
        self.validated = []

    def retrieve_pool(self, query, top_k):
        return list(self.pool), {"num_candidates": len(self.pool), "top_k": top_k}

    def validate_page(self, query, pool_slice):
        self.validated.append(list(pool_slice))
        return [f"kept:{c}" for c in pool_slice]


class _OneShot:
    """A strategy that answers in one pass, like BM25."""

    paginates = False

    def __init__(self):
        self.top_k = None

    def execute(self, query, top_k):
        self.top_k = top_k

        class _R:
            results = ["a", "b"]
            metadata = {"num_chunks": 7}
        return _R()


# ----- what a strategy says about itself -------------------------------------

def test_a_paging_strategy_reports_its_page_size():
    assert SearchRun("Content", _Paging()).page_size == 2


def test_a_one_shot_strategy_has_no_page_size():
    assert SearchRun("BM25", _OneShot()).page_size is None


def test_every_registered_strategy_can_be_asked_for_its_page_size():
    for name, cls in VESPA_STRATEGIES.items():
        run = SearchRun(name, cls(object()))
        assert run.page_size is None or isinstance(run.page_size, int), name


# ----- the first page --------------------------------------------------------

def test_a_paging_strategy_validates_only_the_first_page():
    strategy = _Paging()
    page = SearchRun("Content", strategy).first_page("mold")

    assert isinstance(page, FirstPage)
    assert strategy.validated == [["c0", "c1"]]          # not the whole pool
    assert page.results == ["kept:c0", "kept:c1"]
    assert page.pool == ["c0", "c1", "c2", "c3", "c4"]
    assert page.cursor == 2


def test_a_one_shot_strategy_comes_back_with_no_pool():
    page = SearchRun("BM25", _OneShot()).first_page("Experian")

    assert page.results == ["a", "b"]
    assert page.pool is None
    assert page.cursor == 0


def test_both_shapes_return_the_same_thing():
    a = SearchRun("Content", _Paging()).first_page("mold")
    b = SearchRun("BM25", _OneShot()).first_page("mold")
    assert type(a) is type(b) is FirstPage


def test_the_first_page_reports_how_many_were_validated():
    page = SearchRun("Content", _Paging()).first_page("mold")
    assert page.metadata["validated_count"] == 2
    assert page.metadata["num_candidates"] == 5      # the strategy's own metadata survives


def test_a_pool_shorter_than_a_page_does_not_overrun():
    page = SearchRun("Content", _Paging(pool=["c0"])).first_page("mold")
    assert page.cursor == 1


def test_an_empty_pool_yields_an_empty_first_page():
    page = SearchRun("Content", _Paging(pool=[])).first_page("mold")
    assert page.results == []
    assert page.cursor == 0


def test_the_retrieval_breadth_reaches_the_strategy():
    strategy = _OneShot()
    SearchRun("BM25", strategy).first_page("mold", top_k=250)
    assert strategy.top_k == 250


# ----- going deeper ----------------------------------------------------------

def test_the_next_page_validates_the_slice_it_was_given():
    strategy = _Paging()
    run = SearchRun("Content", strategy)
    run.first_page("mold")

    out = run.next_page("mold", ["c2", "c3"])

    assert strategy.validated == [["c0", "c1"], ["c2", "c3"]]
    assert out == ["kept:c2", "kept:c3"]


def test_going_deeper_does_not_retrieve_again():
    strategy = _Paging()
    run = SearchRun("Content", strategy)
    run.first_page("mold")
    calls_before = len(strategy.validated)

    run.next_page("mold", ["c2", "c3"])

    assert len(strategy.validated) == calls_before + 1   # validated again
    # retrieve_pool would have reset `validated`; it didn't run.


# ----- progress --------------------------------------------------------------

def test_a_paging_run_reports_its_stages_in_order():
    stages = []
    SearchRun("Content", _Paging()).first_page(
        "mold", on_stage=lambda f, t: stages.append((f, t))
    )
    assert [f for f, _ in stages] == sorted(f for f, _ in stages)
    assert stages[-1][0] == 1.0


def test_a_one_shot_run_reports_stages_too():
    stages = []
    SearchRun("BM25", _OneShot()).first_page(
        "mold", on_stage=lambda f, t: stages.append((f, t))
    )
    assert stages[-1][0] == 1.0


def test_a_run_without_a_progress_callback_is_fine():
    SearchRun("Content", _Paging()).first_page("mold")      # must not raise


# ----- resolving a name ------------------------------------------------------

def test_an_unknown_strategy_is_refused():
    with pytest.raises(ValueError, match="Unknown strategy"):
        start_run("Nonexistent Strategy")
