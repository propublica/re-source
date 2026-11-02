"""Reading a Vespa chunk response, end to end."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

from .config import COLBERT_MIN_SCORE, MIN_RESULTS
from .search_types import AtomicSearchResult, SearchResult

logger = logging.getLogger(__name__)


class ChunkSource(Protocol):
    """What reading a chunk response needs from Vespa, and nothing more."""

    def query(self, **kwargs) -> dict: ...

    def get(self, schema: str, doc_id: str) -> dict | None: ...


# Field names retrieved per chunk hit. This module is the parser for this
# select list, so the two belong together.
# parent_ref is intentionally omitted — Vespa reference fields aren't in
# summary. parent_id carries the same value as a plain attribute.
CHUNK_FIELDS = "chunk_id, parent_id, column_name, column_key, text, source_file"


# Max parent ids per batched lookup. One round-trip per 200 parents collapses
# the common case (max_results=200) to a single query, while keeping any one
# YQL request a sane size for find-all sweeps.
_PARENT_FETCH_BATCH = 200


def quote_yql_string(value: str) -> str:
    """Quote a value for use inside a YQL string literal."""
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _parent_id_from_fields(fields: dict) -> str:
    """Parent id for a chunk hit.

    Returns:
        The stored `parent_id` attribute, falling back to deriving it from chunk_id.
    """
    stored = fields.get("parent_id")
    if stored:
        return stored
    return _parent_id_from_chunk(fields.get("chunk_id", ""), fields.get("column_key", ""))


def _parent_id_from_chunk(chunk_id: str, column_key: str) -> str:
    """Derive parent_id from a chunk_id."""
    if not chunk_id or not column_key:
        return chunk_id
    # Strip an optional trailing page suffix (_p0, _p12, …) added to split chunks.
    base = re.sub(r"_p\d+$", "", chunk_id)
    suffix = "_" + column_key
    if base.endswith(suffix):
        return base[:-len(suffix)]
    # Fall back to the un-stripped form (already correct for single-part ids
    # whose column_key legitimately ends in something like "_p3").
    return chunk_id[:-len(suffix)] if chunk_id.endswith(suffix) else chunk_id


def normalize_full_row(value) -> dict:
    """`full_row` is a map<string,string>."""
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        out = {}
        for item in value:
            if isinstance(item, dict) and "key" in item:
                out[str(item["key"])] = item.get("value", "")
        return out
    return {}


def fetch_parent_rows(source: ChunkSource, parent_ids: list[str]) -> dict[str, dict]:
    """parent_id -> full_row, fetched in batched queries."""
    if not parent_ids:
        return {}

    rows: dict[str, dict] = {}
    for i in range(0, len(parent_ids), _PARENT_FETCH_BATCH):
        batch = parent_ids[i : i + _PARENT_FETCH_BATCH]
        ids = ", ".join(f'"{quote_yql_string(p)}"' for p in batch)
        yql = (
            "select parent_id, full_row, source_file from parent "
            f"where parent_id in ({ids})"
        )
        try:
            resp = source.query(yql=yql, ranking="unranked", hits=len(batch), timeout="20s")
        except Exception as e:
            logger.warning(f"[parents] batch fetch failed for {len(batch)} ids: {e}")
            continue
        for hit in resp.get("root", {}).get("children", []) or []:
            f = hit.get("fields", {})
            pid = f.get("parent_id")
            if pid:
                rows[pid] = normalize_full_row(f.get("full_row"))

    missing = [p for p in parent_ids if p not in rows]
    if missing:
        logger.info(
            f"[parents] {len(missing)}/{len(parent_ids)} absent from batch result; "
            "falling back to per-document GET"
        )

        def _one(pid: str) -> tuple[str, dict]:
            fields = source.get("parent", pid) or {}
            return pid, normalize_full_row(fields.get("full_row"))

        with ThreadPoolExecutor(max_workers=16) as ex:
            rows.update(dict(ex.map(_one, missing)))

    return rows


def aggregate_chunks_to_parents(
    source: ChunkSource,
    chunks: list[dict],
    max_results: int = 200,
) -> list[AtomicSearchResult]:
    """Group chunks by parent_id; fetch each parent's full_row from Vespa."""
    per_parent: dict[str, dict] = {}
    for chunk in chunks:
        parent_id = chunk["parent_id"]
        if parent_id not in per_parent:
            per_parent[parent_id] = {
                "parent_id": parent_id,
                "source_file": chunk["source_file"],
                "best_score": chunk["colbert_score"],
                "matching_chunks": [],
            }
        per_parent[parent_id]["matching_chunks"].append({
            "chunk_id": chunk.get("chunk_id"),
            "column_name": chunk["column_name"],
            "column_key": chunk["column_key"],
            "text": chunk["text"],
            "score": chunk["colbert_score"],
        })
        if chunk["colbert_score"] > per_parent[parent_id]["best_score"]:
            per_parent[parent_id]["best_score"] = chunk["colbert_score"]

    ordered = sorted(per_parent.values(), key=lambda x: x["best_score"], reverse=True)[:max_results]

    fetched = fetch_parent_rows(source, [p["parent_id"] for p in ordered])

    return [
        AtomicSearchResult(
            parent_id=parent["parent_id"],
            source_file=parent["source_file"],
            score=parent["best_score"],
            matching_chunks=parent["matching_chunks"],
            full_row=fetched.get(parent["parent_id"], {}),
        )
        for parent in ordered
    ]


def find_relevance_cutoff(scores: list[float], min_results: int = MIN_RESULTS) -> int:
    """Adaptive cutoff for ranked scores, normalized to the top hit so the heuristic is
    scale-invariant.

    Args:
        scores: ranked scores — Qdrant's [0,1] ColBERT or Vespa's unbounded MaxSim.
    """
    if not scores:
        return 0

    top = scores[0]
    if top <= 0:
        return 0
    normalized = [s / top for s in scores]

    threshold_cutoff = 0
    for idx, s in enumerate(normalized):
        if s < COLBERT_MIN_SCORE:
            break
        threshold_cutoff = idx + 1

    if threshold_cutoff == 0:
        return 0

    passing = normalized[:threshold_cutoff]
    # Elbow detection: only fire on a *real* cliff (≥30% drop), and
    # only when we'd still keep ≥5 candidates afterwards. LLM validation
    # downstream means retrieval should over-include rather than risk
    # cutting genuinely relevant results too early.
    if len(passing) > 5:
        diffs = [passing[i] - passing[i + 1] for i in range(len(passing) - 1)]
        max_drop = max(diffs)
        max_drop_idx = diffs.index(max_drop) + 1
        if max_drop >= 0.30 and max_drop_idx >= 5:
            threshold_cutoff = min(threshold_cutoff, max_drop_idx)

    return max(threshold_cutoff, min_results)


def parents_to_search_results(
    parents: list[AtomicSearchResult],
) -> list[SearchResult]:
    """AtomicSearchResult → SearchResult, the shape everything downstream reads."""
    return [
        SearchResult(
            doc_id=p.parent_id,
            text="\n".join(c["text"] for c in p.matching_chunks[:3]),
            source_file=p.source_file,
            fields=p.full_row,
            score=p.score,
            payload={
                "parent_id": p.parent_id,
                "matching_chunks": p.matching_chunks,
                "num_matching_chunks": len(p.matching_chunks),
            },
        )
        for p in parents
    ]


@dataclass
class ChunkReadout:
    """Everything one chunk response yields, read once."""

    results: list[SearchResult]
    parents: list[AtomicSearchResult]
    chunks: list[dict]
    num_candidates: int
    cutoff: int


def _iter_hits(response: dict) -> list[dict]:
    """The hits in a Vespa response. `children` can be absent or null."""
    return response.get("root", {}).get("children", []) or []


def _hit_score(hit: dict, fields: dict, score_feature: str) -> float:
    """The score for one hit, off the named rank feature."""
    return fields.get("matchfeatures", {}).get(score_feature, hit.get("relevance", 0.0))


def read_chunk_response(
    source: ChunkSource,
    response: dict,
    *,
    score_feature: str,
    apply_cutoff: bool = False,
    max_parents: int = 200,
) -> ChunkReadout:
    """Read a chunk response through to parent-level results."""
    hits = _iter_hits(response)
    chunks: list[dict] = []
    for hit in hits:
        fields = hit.get("fields", {})
        chunks.append({
            "chunk_id": fields.get("chunk_id", ""),
            "parent_id": _parent_id_from_fields(fields),
            "column_name": fields.get("column_name"),
            "column_key": fields.get("column_key", ""),
            # A chunk doc can carry a null text field (stale pre-migration
            # docs); a None here would blow up the "\n".join() that builds
            # result.text.
            "text": fields.get("text") or "",
            "source_file": fields.get("source_file"),
            "colbert_score": _hit_score(hit, fields, score_feature),
        })

    if apply_cutoff:
        cutoff = find_relevance_cutoff([c["colbert_score"] for c in chunks])
    else:
        cutoff = len(chunks)
    relevant = chunks[:cutoff]

    parents = aggregate_chunks_to_parents(source, relevant, max_results=max_parents)
    return ChunkReadout(
        results=parents_to_search_results(parents),
        parents=parents,
        chunks=relevant,
        num_candidates=len(hits),
        cutoff=cutoff,
    )


def best_score_by_parent(response: dict, *, score_feature: str) -> dict[str, float]:
    """Best score per parent, for a response that carries only ids and scores."""
    best: dict[str, float] = {}
    for hit in _iter_hits(response):
        fields = hit.get("fields", {})
        score = _hit_score(hit, fields, score_feature)
        pid = _parent_id_from_fields(fields)
        if score > best.get(pid, float("-inf")):
            best[pid] = score
    return best
