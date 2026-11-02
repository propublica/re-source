"""Pydantic request/response models for the search API."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

# Shared with the Streamlit door, so the two cannot drift.
from re_source.config import MAX_QUERY_CHARS
from re_source.vespa_strategies import DEFAULT_STRATEGY


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=MAX_QUERY_CHARS)
    strategy: str = DEFAULT_STRATEGY
    top_k: int = Field(default=100, ge=1, le=2000)


class SearchResultOut(BaseModel):
    doc_id: str
    text: Optional[str] = None
    source_file: str
    fields: dict[str, Any]
    score: float
    payload: Optional[dict[str, Any]] = None


class SearchResponse(BaseModel):
    approach_name: str
    elapsed_seconds: float
    metadata: dict[str, Any]
    results: list[SearchResultOut]
