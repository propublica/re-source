"""Putting several retrieval lanes' results into one ranked pool."""

from __future__ import annotations

from dataclasses import dataclass, field

from .field_value import is_blank_value, without_label
from .search_types import SearchResult


@dataclass
class RankedPool:
    """The union, ranked, with what it took to get there."""

    results: list[SearchResult]
    metadata: dict = field(default_factory=dict)


def chunk_count(result: SearchResult) -> int:
    return len((result.payload or {}).get("matching_chunks") or [])


def is_contentless(result: SearchResult) -> bool:
    """True when every matched chunk is a question label over a blank value."""
    chunks = (result.payload or {}).get("matching_chunks") or []
    if not chunks:
        return not (result.text or "").strip()
    for c in chunks:
        answer = without_label(c.get("text") or "", c.get("column_name") or "")
        if not is_blank_value(answer.strip()):
            return False
    return True


def tail_chunk_ids(union: dict[str, SearchResult], needs_score) -> list[str]:
    """Chunk ids for the union members that still need a comparable score."""
    ids = []
    for pid in needs_score:
        for c in (union[pid].payload or {}).get("matching_chunks") or []:
            if c.get("chunk_id"):
                ids.append(c["chunk_id"])
    return ids


def rank(lane_results: dict[str, list[SearchResult]], *, comparable_lanes,
         rescore=None) -> RankedPool:
    """One ranked pool from several lanes' results."""
    comparable_lanes = set(comparable_lanes)

    union: dict[str, SearchResult] = {}
    comparable: dict[str, float] = {}
    lanes_hit: dict[str, set] = {}

    for lane_name, results in lane_results.items():
        for r in results:
            pid = r.doc_id
            lanes_hit.setdefault(pid, set()).add(lane_name)
            current = union.get(pid)
            if current is None or chunk_count(r) > chunk_count(current):
                union[pid] = r          # richest chunk context wins, for display
            if lane_name in comparable_lanes and r.score > comparable.get(pid, float("-inf")):
                comparable[pid] = r.score

    tail = [pid for pid in union if pid not in comparable]
    rescored = {}
    if tail and rescore is not None:
        rescored = rescore(tail_chunk_ids(union, tail)) or {}

    # Anything with no comparable score of its own sits just under everything
    # that has one — kept, but never ahead of a row we can actually compare.
    floor = min(comparable.values()) if comparable else 0.0
    for pid, r in union.items():
        if pid in comparable:
            r.score = comparable[pid]
        elif pid in rescored:
            r.score = rescored[pid]
        else:
            r.score = floor - 1.0
        if r.payload is None:
            r.payload = {}
        r.payload["lanes"] = sorted(lanes_hit.get(pid, ()))

    ranked = sorted(union.values(), key=lambda r: r.score, reverse=True)

    before_filter = len(ranked)
    ranked = [r for r in ranked if not is_contentless(r)]

    return RankedPool(
        results=ranked,
        metadata={
            "union_size": len(union),
            "per_lane_counts": {ln: len(rs) for ln, rs in lane_results.items()},
            "tail_size": len(tail),
            "tail_reranked": len(rescored),
            "contentless_dropped": before_filter - len(ranked),
            "pool_size": len(ranked),
        },
    )
