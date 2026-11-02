"""Where a record came from: its dataset's label, a link back to it, and the
group a deployment files it under."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from threading import Lock

import duckdb

from .config import DUCKDB_PATH

logger = logging.getLogger(__name__)


# This is application metadata, not a source dataset. Keep the name here as the
# single contract shared by the reader, writer, and document loader.
DATASET_CATALOG_TABLE = "dataset_catalog"

# Table names owned by something other than a file import -- a sync that will
# overwrite them on its next run. Listed as regexes rather than inferred,
# because a table has to be refused as an upload destination from the moment
# the sync could create it, which is before any catalog entry for it exists.
READ_ONLY_TABLE_PATTERNS = tuple(
    p.strip() for p in os.getenv("READ_ONLY_TABLE_PATTERNS", "").split(",") if p.strip()
)


_cache_lock = Lock()
_cached_path: Path | None = None
_cached_signature: tuple[int, int, int, int] | None = None
_cached_metadata: dict[str, dict[str, str]] = {}
_failed_signature: tuple[int, int, int, int] | None = None


def _stat_pair(path: Path) -> tuple[int, int]:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return (0, 0)
    return (stat.st_mtime_ns, stat.st_size)


def _database_signature(path: Path) -> tuple[int, int, int, int] | None:
    """A cheap change token for the database and an uncheckpointed WAL."""
    if not path.exists():
        return None
    wal = Path(f"{path}.wal")
    return (*_stat_pair(path), *_stat_pair(wal))


def _read_catalog(path: Path) -> dict[str, dict[str, str]]:
    conn = duckdb.connect(str(path), read_only=True)
    try:
        exists = conn.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_name = ?",
            [DATASET_CATALOG_TABLE],
        ).fetchone()[0]
        if not exists:
            return {}
        columns = {
            name for (name,) in conn.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name = ?",
                [DATASET_CATALOG_TABLE],
            ).fetchall()
        }
        # A catalog written before groups existed has no such column until the
        # next upsert adds it.
        group = "dataset_group" if "dataset_group" in columns else "NULL"
        rows = conn.execute(
            f'SELECT dataset_id, label, record_url, {group} FROM "{DATASET_CATALOG_TABLE}"'
        ).fetchall()
    finally:
        conn.close()

    return {
        dataset_id: {
            **({"label": label} if label else {}),
            **({"record_url": record_url} if record_url else {}),
            **({"group": group} if group else {}),
        }
        for dataset_id, label, record_url, group in rows
    }


def dataset_metadata() -> dict[str, dict[str, str]]:
    """Return the dataset catalog, refreshing when DuckDB changes.

    A sync may briefly hold DuckDB's cross-process write lock. During that
    window readers keep the last good snapshot and retry on their next lookup.
    """
    global _cached_path, _cached_signature, _cached_metadata, _failed_signature

    path = Path(DUCKDB_PATH)
    signature = _database_signature(path)
    if signature is None:
        return {}

    with _cache_lock:
        if path == _cached_path and signature == _cached_signature:
            return _cached_metadata
        try:
            metadata = _read_catalog(path)
        except duckdb.Error as exc:
            if signature != _failed_signature:
                logger.warning(
                    "Dataset catalog unavailable; using stable ids until the "
                    "next successful read: %s",
                    exc,
                )
                _failed_signature = signature
            if path == _cached_path:
                return _cached_metadata
            return {}
        _cached_path = path
        _cached_signature = signature
        _cached_metadata = metadata
        _failed_signature = None
        return metadata


def upsert_dataset_metadata(conn, entries: dict[str, dict[str, str | None]]) -> int:
    """Merge dataset metadata into DuckDB, preserving absent prior labels and groups."""
    if not entries:
        return 0

    rows = [
        (
            dataset_id,
            entry.get("label") or None,
            entry.get("record_url") or None,
            entry.get("group") or None,
        )
        for dataset_id, entry in entries.items()
    ]
    conn.execute("BEGIN TRANSACTION")
    try:
        conn.execute(
            f'CREATE TABLE IF NOT EXISTS "{DATASET_CATALOG_TABLE}" ('
            "dataset_id VARCHAR PRIMARY KEY, label VARCHAR, record_url VARCHAR, "
            "dataset_group VARCHAR)"
        )
        conn.execute(
            f'ALTER TABLE "{DATASET_CATALOG_TABLE}" '
            "ADD COLUMN IF NOT EXISTS dataset_group VARCHAR"
        )
        conn.executemany(
            f'INSERT INTO "{DATASET_CATALOG_TABLE}" '
            "(dataset_id, label, record_url, dataset_group) VALUES (?, ?, ?, ?) "
            "ON CONFLICT (dataset_id) DO UPDATE SET "
            f'label = COALESCE(excluded.label, "{DATASET_CATALOG_TABLE}".label), '
            "record_url = excluded.record_url, "
            f'dataset_group = COALESCE(excluded.dataset_group, "{DATASET_CATALOG_TABLE}".dataset_group)',
            rows,
        )
    except Exception:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")
    reload_metadata()
    return len(rows)


def delete_dataset_metadata(conn, dataset_ids) -> int:
    """Remove catalog entries for datasets that no longer exist."""
    ids = list(dataset_ids)
    exists = conn.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = ?",
        [DATASET_CATALOG_TABLE],
    ).fetchone()[0]
    if not ids or not exists:
        return 0
    placeholders = ", ".join("?" * len(ids))
    (deleted,) = conn.execute(
        f'DELETE FROM "{DATASET_CATALOG_TABLE}" WHERE dataset_id IN ({placeholders})',
        ids,
    ).fetchone()
    reload_metadata()
    return deleted


def _entry(dataset_id: str) -> dict[str, str]:
    entry = dataset_metadata().get(dataset_id)
    return entry if isinstance(entry, dict) else {}


def dataset_label(dataset_id: str) -> str:
    """Human-readable dataset name, falling back to the stable id."""
    return _entry(dataset_id).get("label") or dataset_id


def dataset_group(dataset_id: str) -> str | None:
    """The group a deployment filed this dataset under, or None."""
    return _entry(dataset_id).get("group") or None


def record_url(dataset_id: str, record_id: str) -> str | None:
    """Link to the record where it originated, or None if there is none."""
    template = _entry(dataset_id).get("record_url")
    if not (template and dataset_id and record_id):
        return None
    try:
        return template.format(dataset_id=dataset_id, record_id=record_id)
    except (KeyError, IndexError):
        # A template naming a field we do not supply is a config error, but not
        # one worth failing a search over: the card simply gets no link.
        return None


def is_externally_owned(name: str) -> bool:
    """True if this table is maintained by something other than file import."""
    # Case-insensitive throughout because DuckDB is: it resolves table
    # identifiers without regard to case even when quoted, so a case variant of
    # a synced table name is the same table and must be refused the same way.
    if any(re.match(p, name, re.IGNORECASE) for p in READ_ONLY_TABLE_PATTERNS):
        return True
    folded = name.casefold()
    return any(folded == known.casefold() for known in dataset_metadata())


def reload_metadata() -> None:
    """Drop the process-local snapshot after an in-process catalog write."""
    global _cached_path, _cached_signature, _cached_metadata, _failed_signature
    with _cache_lock:
        _cached_path = None
        _cached_signature = None
        _cached_metadata = {}
        _failed_signature = None
