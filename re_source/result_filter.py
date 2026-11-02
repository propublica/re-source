"""Narrowing a result set — which rows a reporter is actually looking at."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from typing import MutableMapping

from . import config
from .datasets import dataset_group, dataset_label
from .result_card import submitted_date

# The widget keys this filter owns. Named here because `reconcile` writes one
# of them and the page draws a widget under it, and those two have to be the
# same string or a reconciled selection is silently discarded.
#
# re_source/search_session.FILTER_KEYS lists them again, for a different job: what
# "Reset filters" clears. Consolidating would mean that generic state module
# importing this domain one, which is the wrong direction for three strings —
# so a test pins the two lists against each other instead.
DATASETS_KEY = "filter-datasets"
DATES_KEY = "filter-dates"
OFFERED_KEY = "filter-datasets-offered"
# One checkbox per group exclusion, keyed by its group.
EXCLUDE_KEY_PREFIX = "filter-exclude::"


@dataclass(frozen=True)
class FilterOffer:
    """What there is to narrow by, given a set of results."""

    datasets: tuple[str, ...]


@dataclass(frozen=True)
class Narrowing:
    """What a reporter chose to narrow by."""

    datasets: tuple[str, ...] | None = None
    dates: tuple[date, date] | None = None


@dataclass(frozen=True)
class Narrowed:
    """The rows that survived, and what it cost to get there."""

    results: list
    undated: int


@dataclass(frozen=True)
class DateChoice:
    """What a date widget's current value amounts to.

    `incomplete` is the case worth having a name for: a start date picked and
    no end yet. It is not a range, so nothing can be filtered by it — but it
    looks to the reporter exactly like a range they have set, which is why the
    screen needs to be able to say so.
    """

    dates: tuple[date, date] | None = None
    incomplete: bool = False


@dataclass(frozen=True)
class GroupExclusion:
    """A checkbox that hides every dataset filed under one catalog group."""

    label: str
    group: str

    @property
    def key(self) -> str:
        return f"{EXCLUDE_KEY_PREFIX}{self.group}"


@lru_cache(maxsize=None)
def _load_group_exclusions(module_name: str):
    module = importlib.import_module(module_name)
    resolver = getattr(module, "group_exclusions", None)
    if not callable(resolver):
        raise ValueError(
            f"FILTER_POLICY_MODULE {module_name!r} must define group_exclusions()"
        )
    return resolver


def group_exclusions() -> list[GroupExclusion]:
    """The group checkboxes on offer, from ``FILTER_POLICY_MODULE``."""
    # Public Re-Source has none; a deployment names its own groups.
    module_name = config.FILTER_POLICY_MODULE.strip()
    if not module_name:
        return []
    return [GroupExclusion(label, group) for label, group in _load_group_exclusions(module_name)()]


def without_groups(results, groups) -> list:
    """`results` minus every row whose dataset is filed under one of `groups`."""
    hidden = set(groups)
    if not hidden:
        return list(results)
    return [r for r in results if dataset_group(r.source_file) not in hidden]


def date_choice(value) -> DateChoice:
    """Read a date-range widget value.

    A range-mode `st.date_input` hands back 0, 1 or 2 dates depending on how
    far through the selection the reporter is, and only 2 is a range. The
    caller used to collapse everything else to None inline, which was right
    about the filtering and silent about the middle case: a reporter who had
    picked a start date and not an end saw an unfiltered list while the help
    text beside it promised undated rows were being excluded.
    """
    if not isinstance(value, (tuple, list)):
        return DateChoice()
    picked = [v for v in value if v is not None]
    if len(picked) == 2:
        return DateChoice(dates=(picked[0], picked[1]))
    return DateChoice(incomplete=len(picked) == 1)


def offer(results) -> FilterOffer:
    """What these results can be narrowed by."""
    return FilterOffer(datasets=tuple(sorted({dataset_label(r.source_file) for r in results})))


def widen(selected, previously_offered, now_offered) -> tuple[str, ...]:
    """The dataset selection, with datasets seen for the first time added to it."""
    first_seen = set(now_offered) - set(previously_offered)
    keep = set(selected) | first_seen
    return tuple(dataset for dataset in now_offered if dataset in keep)


def reconcile(state: MutableMapping, results) -> FilterOffer:
    """The datasets on offer for `results`, with the stored selection brought into
    line with them. Records what was offered, for the next call to compare.

    Call this *before* the datasets widget is drawn. That ordering is the whole
    point: the widget reads its own key as its state, so writing the reconciled
    selection first means it picks that up, and writing it afterwards means
    fighting a widget that has already rendered. A plain mapping goes in —
    `st.session_state` in the app, a dict in tests — following the convention
    SearchSession and ExportSelection already set, so the ordering and the key
    names are assertable without driving four widgets.

    They were not, before. `widen` had nine tests proving a function nobody
    could get wrong, and the read-widen-write-back that wraps it was nine lines
    inside a render function with no test surface at all.
    """
    on_offer = offer(results)
    stored = state.get(DATASETS_KEY)
    if stored is not None:
        state[DATASETS_KEY] = list(
            widen(stored, state.get(OFFERED_KEY, ()), on_offer.datasets)
        )
    state[OFFERED_KEY] = list(on_offer.datasets)
    return on_offer


def apply(results, narrowing: Narrowing) -> Narrowed:
    """The rows a reporter is looking at, and how many a date bound excluded
    for having no usable date of their own."""
    rows = list(results)

    if narrowing.datasets is not None:
        allowed = set(narrowing.datasets)
        rows = [r for r in rows if dataset_label(r.source_file) in allowed]

    undated = 0
    if narrowing.dates is not None:
        start, end = narrowing.dates
        kept = []
        for r in rows:
            when = submitted_date(r)
            if when is None:
                undated += 1
                continue
            if start <= when <= end:
                kept.append(r)
        rows = kept

    return Narrowed(results=rows, undated=undated)
