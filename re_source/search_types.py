"""Lightweight, dependency-free result dataclasses shared by the Qdrant
(`re_source/search.py`) and Vespa (`re_source/vespa_strategies.py`) search paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class SearchResult:
    """A single search result."""
    doc_id: str
    text: str
    source_file: str
    fields: dict[str, str]
    score: float
    # Extracted semantic fields
    payload: Optional[dict] = None


@dataclass
class AtomicSearchResult:
    """Result from atomic chunk search, aggregated to parent row."""
    parent_id: str
    source_file: str
    score: float
    matching_chunks: list[dict]  # Which fields/chunks matched
    full_row: dict  # All fields from parent for context
