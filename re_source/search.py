"""Backward-compatible re-export of the search result dataclasses."""

from .search_types import AtomicSearchResult, SearchResult

__all__ = ["SearchResult", "AtomicSearchResult"]
