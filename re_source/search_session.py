"""One running search: its results, how deep into the pool it has gone, and
what has already been recorded as disclosed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, MutableMapping

# The filter widgets' own keys. Dropping a widget key is how a filter goes back
# to its default: the next render sees no stored value and falls back to "every
# dataset, no date bound".
FILTER_KEYS = (
    "filter-datasets",
    "filter-dates",
    # What the dataset menu offered last time it was drawn, so a dataset arriving
    # via "Load more" can be told from one the reporter deselected.
    "filter-datasets-offered",
)
# Filter widgets whose keys are made per deployment: one group-exclusion checkbox
# per group (result_filter.EXCLUDE_KEY_PREFIX).
FILTER_KEY_PREFIXES = ("filter-exclude::",)

# Everything describing one search, and nothing else. Cleared by "Clear
# search", by the wordmark, and by switching strategies.
SEARCH_KEYS = (
    "vespa_results",
    "vespa_metadata",
    "vespa_elapsed",
    "vespa_query",
    "vespa_strategy",
    "vespa_pool",
    "vespa_cursor",
    "audit_seen_records",
    "audit_page",
    *FILTER_KEYS,
)


@dataclass(frozen=True)
class ActiveSearch:
    """What is currently on screen, and what is left to go deeper into."""

    results: list
    elapsed: float | None
    metadata: dict
    query: str
    strategy: str
    pool: list | None
    cursor: int


@dataclass(frozen=True)
class PagePlan:
    """What one 'Load more' click would do, worked out before it happens."""

    cursor: int
    step: int
    remaining: int
    pool_slice: list


class SearchSession:
    """The current search, over one state mapping."""

    def __init__(self, state: MutableMapping) -> None:
        self._state = state

    # ----- reading ---------------------------------------------------------

    def active(self) -> ActiveSearch | None:
        """The search on screen, or None if nothing has run yet."""
        results = self._state.get("vespa_results")
        if results is None:
            return None
        return ActiveSearch(
            results=results,
            elapsed=self._state.get("vespa_elapsed"),
            metadata=self._state.get("vespa_metadata") or {},
            query=self._state.get("vespa_query", ""),
            strategy=self._state.get("vespa_strategy", ""),
            pool=self._state.get("vespa_pool"),
            cursor=self._state.get("vespa_cursor", 0),
        )

    def page_plan(self, page_size: int | None) -> PagePlan | None:
        """What the next page would be, or None when there isn't one."""
        if page_size is None:
            return None
        pool = self._state.get("vespa_pool")
        if pool is None:
            return None
        cursor = self._state.get("vespa_cursor", 0)
        remaining = len(pool) - cursor
        if remaining <= 0:
            return None
        step = min(page_size, remaining)
        return PagePlan(
            cursor=cursor,
            step=step,
            remaining=remaining,
            pool_slice=pool[cursor:cursor + step],
        )

    # ----- writing ---------------------------------------------------------

    def start(self, *, results, metadata, elapsed, query, strategy,
              pool=None, cursor=0) -> None:
        """Replace whatever was on screen with a fresh search."""
        self._state["vespa_results"] = results
        self._state["vespa_metadata"] = metadata
        self._state["vespa_elapsed"] = elapsed
        self._state["vespa_query"] = query
        self._state["vespa_strategy"] = strategy
        self._state["vespa_pool"] = pool
        self._state["vespa_cursor"] = cursor

    def extend(self, more: list, step: int) -> int:
        """Append a validated page and advance the cursor. Returns the cursor."""
        results = (self._state.get("vespa_results") or []) + list(more)
        self._state["vespa_results"] = results
        self._state["vespa_cursor"] = self._state.get("vespa_cursor", 0) + step
        metadata = self._state.get("vespa_metadata") or {}
        self._state["vespa_metadata"] = {**metadata, "validated_count": len(results)}
        return self._state["vespa_cursor"]

    def clear(self) -> None:
        """Drop the search so the controls come back empty."""
        for key in SEARCH_KEYS:
            self._state.pop(key, None)
        self._drop_prefixed_filters()

    def clear_filters(self) -> None:
        """Put the filters back to showing everything."""
        for key in FILTER_KEYS:
            self._state.pop(key, None)
        self._drop_prefixed_filters()

    def _drop_prefixed_filters(self) -> None:
        for key in [k for k in self._state if str(k).startswith(FILTER_KEY_PREFIXES)]:
            self._state.pop(key, None)

    # ----- session-scoped, deliberately outliving a search -----------------

    def trace_session_id(self, new_id: Callable[[], str]) -> str:
        """Stable per-session id so a user's queries group in Langfuse."""
        if "langfuse_session_id" not in self._state:
            self._state["langfuse_session_id"] = new_id()
        return self._state["langfuse_session_id"]

    def remembered_strategy(self) -> str | None:
        """The strategy the user last chose."""
        return self._state.get("search_strategy_choice")

    def remember_strategy(self, name: str) -> None:
        self._state["search_strategy_choice"] = name

    # ----- what has been disclosed -----------------------------------------

    def begin_audit_page(self, *, new_search: bool) -> int:
        """Open a page for audit purposes and return its number."""
        if new_search:
            self._state["audit_seen_records"] = set()
            self._state["audit_page"] = 1
        else:
            self._state["audit_page"] = self._state.get("audit_page", 1) + 1
        return self._state["audit_page"]

    def unseen(self, refs: list[str]) -> list[str]:
        """The refs not yet recorded in this search, marking them as recorded."""
        seen = self._state.setdefault("audit_seen_records", set())
        fresh = [r for r in refs if r not in seen]
        seen.update(fresh)
        return fresh
