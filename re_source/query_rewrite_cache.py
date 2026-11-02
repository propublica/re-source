"""Persistent cache for LLM query rewrites (HyDE, expansion, decomposition)."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Callable, TypeVar

from .config import DATA_DIR

logger = logging.getLogger(__name__)

T = TypeVar("T")

CACHE_DIR = Path(os.getenv("QUERY_REWRITE_CACHE_DIR", str(DATA_DIR / "query_rewrite_cache")))


def cache_enabled() -> bool:
    return os.getenv("QUERY_REWRITE_CACHE", "1").lower() not in ("0", "false", "no")


def _key(kind: str, prompt: str, model_id: str) -> str:
    """Hash of what actually determines the answer."""
    h = hashlib.sha256()
    for part in (kind, model_id, prompt):
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _path(key: str) -> Path:
    return CACHE_DIR / f"{key}.json"


def _read(key: str):
    try:
        with open(_path(key), encoding="utf-8") as f:
            return json.load(f)["result"]
    except (FileNotFoundError, KeyError, ValueError, OSError):
        return None


def _write(key: str, kind: str, model_id: str, result) -> None:
    """Atomic write, so a concurrent reader never sees a half-written entry and
    an interrupted process can't leave a corrupt one."""
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {"kind": kind, "model_id": model_id, "result": result},
            ensure_ascii=False,
        )
        fd, tmp = tempfile.mkstemp(dir=CACHE_DIR, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(payload)
            os.replace(tmp, _path(key))
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
    except OSError as e:
        logger.debug(f"[rewrite-cache] write failed for {kind}: {e}")


def cached_rewrite(
    kind: str,
    prompt: str,
    model_id: str,
    compute: Callable[[], T],
    *,
    valid: Callable[[T], bool] | None = None,
) -> T:
    """Return a cached rewrite for this exact prompt, or compute and store it.

    Args:
        kind: rewrite type, for readability in stored entries ("hyde", ...).
        prompt: the full prompt text — the cache key, so any prompt edit
            invalidates its own entries.
        model_id: provider and model, so switching either invalidates too.
        compute: called on a miss. Its return value must be JSON-serialisable.
        valid: optional shape check, applied to a cached entry before it is
            returned and to a computed value before it is stored. A caller's
            own checks live inside `compute`, which a cache hit skips — so
            without this a malformed entry is served forever.
    """
    if not cache_enabled():
        return compute()

    key = _key(kind, prompt, model_id)
    hit = _read(key)
    if hit is not None:
        if valid is None or valid(hit):
            logger.debug(f"[rewrite-cache] hit {kind}")
            return hit
        logger.warning(
            f"[rewrite-cache] discarding malformed {kind} entry "
            f"({type(hit).__name__}); recomputing"
        )
        _discard(key)

    result = compute()
    if result is not None and (valid is None or valid(result)):
        _write(key, kind, model_id, result)
    return result


def _discard(key: str) -> None:
    """Remove one entry. Best-effort: failing to delete it only means the next
    call re-does the same work, which the caller already survived once."""
    try:
        _path(key).unlink(missing_ok=True)
    except OSError as e:
        logger.debug(f"[rewrite-cache] could not discard {key}: {e}")


def clear() -> int:
    """Delete every cached rewrite.

    Returns:
        How many entries were removed.
    """
    if not CACHE_DIR.exists():
        return 0
    n = 0
    for path in CACHE_DIR.glob("*.json"):
        try:
            path.unlink()
            n += 1
        except OSError:
            pass
    return n
