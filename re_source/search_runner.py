"""Running a Vespa search strategy by name — once, or a page at a time."""

from __future__ import annotations

import threading
from contextlib import nullcontext
from dataclasses import dataclass

from re_source.vespa_client import get_vespa_client
from re_source.vespa_strategies import DEFAULT_TOP_K, VESPA_STRATEGIES, StrategyResult

# Bound concurrent LLM-heavy runs so parallel API requests don't oversubscribe
# the LLM validation backend. Vespa retrieval itself is server-side.
#
# This bounds callers of run_search, which is not the same as bounding the
# process: a Streamlit search runs its own LLM validation (VALIDATION_WORKERS
# batches at a time) outside this semaphore, so concurrent browser sessions add
# to whatever the API is doing rather than queueing behind it. Extending the
# bound to the UI is a user-visible change — a second reporter's search would
# wait behind a progress bar that doesn't say so — and is deliberately not done
# here.
_run_semaphore = threading.Semaphore(4)


def build_strategy(strategy: str):
    """Build a strategy from its registry name, on the shared Vespa client.

    Raises:
        ValueError: if `strategy` is not a known strategy name.
    """
    if strategy not in VESPA_STRATEGIES:
        raise ValueError(
            f"Unknown strategy: {strategy!r}. "
            f"Available: {', '.join(VESPA_STRATEGIES)}"
        )
    return VESPA_STRATEGIES[strategy](get_vespa_client())


def run_search(strategy: str, query: str, top_k: int = 100) -> StrategyResult:
    """Run one Vespa strategy by its registry name.

    Args:
        strategy: a key of VESPA_STRATEGIES.
        query: the search query.
        top_k: retrieval breadth (and, for LLM strategies, how many get verified).

    Raises:
        ValueError: if `strategy` is not a known strategy name.
    """
    strat = build_strategy(strategy)
    with _run_semaphore:
        return strat.execute(query=query, top_k=top_k)


@dataclass(frozen=True)
class FirstPage:
    """What a search yields before anyone asks for more."""

    results: list
    metadata: dict
    pool: list | None
    cursor: int


class SearchRun:
    """One strategy, ready to run and — if it pages — to go deeper."""

    def __init__(self, name: str, strategy, attribution=None) -> None:
        self.name = name
        self._strategy = strategy
        self._attribution = attribution or (lambda **_: nullcontext())

    @property
    def paginates(self) -> bool:
        """Whether this strategy offers a next page. It declares this itself."""
        return getattr(self._strategy, "paginates", False)

    @property
    def page_size(self) -> int | None:
        """Candidates validated per page, or None for a strategy that doesn't page."""
        return self._strategy.PAGE_SIZE if self.paginates else None

    def first_page(self, query: str, *, top_k: int = DEFAULT_TOP_K,
                   on_stage=None) -> FirstPage:
        """Run `query` and return what to show first."""
        with self._attribution(strategy=self.name, top_k=int(top_k)):
            if not self.paginates:
                if on_stage:
                    on_stage(0.4, "Searching…")
                result = self._strategy.execute(query=query, top_k=int(top_k))
                if on_stage:
                    on_stage(1.0, "Done")
                return FirstPage(results=result.results, metadata=result.metadata,
                                 pool=None, cursor=0)

            if on_stage:
                on_stage(0.3, "Retrieving candidates…")
            pool, meta = self._strategy.retrieve_pool(query=query, top_k=int(top_k))
            if on_stage:
                on_stage(0.7, "Checking candidates against your query…")
            page_size = self._strategy.PAGE_SIZE
            page = self._strategy.validate_page(
                query=query, pool_slice=pool[:page_size]
            )
            if on_stage:
                on_stage(1.0, "Done")
        return FirstPage(
            results=page,
            metadata={**meta, "validated_count": len(page)},
            pool=pool,
            cursor=min(page_size, len(pool)),
        )

    def next_page(self, query: str, pool_slice: list) -> list:
        """Validate one more slice of an already-retrieved pool."""
        with self._attribution(strategy=self.name, load_more=True):
            return self._strategy.validate_page(query=query, pool_slice=pool_slice)


def start_run(strategy: str, *, attribution=None) -> SearchRun:
    """A SearchRun for a registry name.

    Raises:
        ValueError: if `strategy` is not a known strategy name.
    """
    return SearchRun(strategy, build_strategy(strategy), attribution=attribution)
