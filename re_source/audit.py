"""Audit trail: which user saw or exported which records."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from re_source import config

logger = logging.getLogger(__name__)

# 2: the search query is no longer recorded.
SCHEMA_VERSION = 2

EVENT_EXPORT = "export"
EVENT_RESULTS_VIEWED = "results_viewed"

SURFACE_STREAMLIT = "streamlit"
SURFACE_API = "api"
# Offline batch runs (scripts/), which disclose records to whoever runs them.
SURFACE_CLI = "cli"


def record_ref(result) -> str:
    """Stable `source_file::record_id` reference for one search result."""
    fields = getattr(result, "fields", None) or {}
    ident = fields.get("_record_id") or getattr(result, "doc_id", None) or "unknown"
    return f"{getattr(result, 'source_file', 'unknown')}::{ident}"


def record_refs(results) -> list[str]:
    """`record_ref` over an iterable of results, order preserved."""
    return [record_ref(r) for r in results]


def build_event(
    event: str,
    *,
    actor: str,
    record_ids: list[str],
    session_id: str | None = None,
    surface: str = SURFACE_STREAMLIT,
    strategy: str | None = None,
    filename: str | None = None,
    page: int | None = None,
    now: datetime | None = None,
) -> dict:
    """Assemble one audit event. Pure — no I/O, so tests can assert on shape."""
    stamp = now or datetime.now(timezone.utc)
    ids = list(record_ids)
    return {
        "schema_version": SCHEMA_VERSION,
        "event_id": uuid.uuid4().hex,
        "ts": stamp.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "event": event,
        "actor": actor,
        "session_id": session_id,
        "surface": surface,
        "strategy": strategy,
        "record_ids": ids,
        "record_count": len(ids),
        "filename": filename,
        "page": page,
    }


def s3_key(evt: dict, prefix: str) -> str:
    """Hive-partitioned key so Athena/DuckDB can prune by day and event type
    without a catalog.
    """
    ts = evt["ts"]                      # 2026-09-10T18:22:04.113Z
    day = ts[:10]
    epoch_ms = int(
        datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp() * 1000
    )
    # epoch-first so keys sort chronologically; uuid slice so they never collide.
    name = f"{epoch_ms}-{evt['event_id'][:8]}.jsonl"
    parts = [p for p in (prefix, f"dt={day}", f"event={evt['event']}", name) if p]
    return "/".join(parts)


class _S3Sink:
    """Writes one JSONL object per event."""

    def __init__(self, bucket: str, prefix: str, region: str | None):
        self.bucket = bucket
        self.prefix = prefix
        self.region = region
        self._client = None

    def client(self):
        if self._client is None:
            import boto3  # local import: only the S3 sink pays for it

            self._client = boto3.client("s3", region_name=self.region)
        return self._client

    def write(self, evt: dict) -> None:
        self.client().put_object(
            Bucket=self.bucket,
            Key=s3_key(evt, self.prefix),
            Body=(json.dumps(evt) + "\n").encode("utf-8"),
            ContentType="application/x-ndjson",
        )


class _FileSink:
    """Appends to a local JSONL file."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def write(self, evt: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(evt) + "\n")


def _active_sink():
    """Resolve the configured sink, or None when auditing is off."""
    sink = (config.AUDIT_SINK or "").lower()
    if sink == "off":
        return None
    if sink == "s3":
        if not config.AUDIT_S3_BUCKET:
            # Misconfiguration, not an outage — say so once, loudly, and let
            # the caller's event fall through to the spool rather than vanish.
            raise RuntimeError("AUDIT_SINK=s3 but AUDIT_S3_BUCKET is empty")
        return _S3Sink(config.AUDIT_S3_BUCKET, config.AUDIT_S3_PREFIX, config.AUDIT_S3_REGION)
    return _FileSink(config.AUDIT_SPOOL_PATH)


def _spool(evt: dict) -> None:
    """Last resort for an event the configured sink refused."""
    _FileSink(config.AUDIT_SPOOL_PATH).write(evt)


def log_event(
    event: str,
    *,
    actor: str,
    record_ids: list[str],
    session_id: str | None = None,
    surface: str = SURFACE_STREAMLIT,
    strategy: str | None = None,
    filename: str | None = None,
    page: int | None = None,
) -> dict | None:
    """Record one audit event. Never raises."""
    try:
        evt = build_event(
            event,
            actor=actor,
            record_ids=record_ids,
            session_id=session_id,
            surface=surface,
            strategy=strategy,
            filename=filename,
            page=page,
        )
    except Exception:
        logger.exception("Audit event could not be built; nothing recorded")
        return None

    try:
        sink = _active_sink()
    except Exception:
        logger.exception("Audit sink unavailable; spooling event %s", evt["event_id"])
        _try_spool(evt)
        return evt

    if sink is None:
        return None

    try:
        sink.write(evt)
    except Exception:
        logger.exception("Audit write failed; spooling event %s", evt["event_id"])
        _try_spool(evt)
    return evt


def _try_spool(evt: dict) -> None:
    """Spool, and if even that fails, at least get the event into the log
    stream — the container's stdout is then the only surviving copy."""
    try:
        _spool(evt)
    except Exception:
        logger.error("Audit spool failed; unrecorded event: %s", json.dumps(evt))
