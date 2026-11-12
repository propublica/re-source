"""Records leaving the system in a workbook, and the event that accounts for it."""

from __future__ import annotations

import getpass
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from re_source import audit
from re_source.export import export_by_query_xlsx, export_vespa_xlsx


def default_actor() -> str:
    """Who an offline run is attributed to: $AUDIT_ACTOR, else the OS user."""
    return os.environ.get("AUDIT_ACTOR") or getpass.getuser()


def workbook_filename() -> str:
    """`resource_{timestamp}.xlsx` — the name a reporter receives."""
    # No query in the name: the filename is recorded in the audit log, and a
    # search can name the very person a tip is about.
    return f"resource_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"


@dataclass
class Disclosure:
    """A workbook that has not been handed over yet, and what recording it takes."""

    data: bytes
    filename: str
    record_ids: list[str]
    actor: str
    surface: str
    query: str | None = None
    strategy: str | None = None
    session_id: str | None = None

    def record(self) -> dict | None:
        """Record that these records left the system."""
        return audit.log_event(
            audit.EVENT_EXPORT,
            actor=self.actor,
            record_ids=self.record_ids,
            session_id=self.session_id,
            surface=self.surface,
            query=self.query,
            strategy=self.strategy,
            filename=self.filename,
        )

    def write(self, path: Path | str) -> dict | None:
        """Write the workbook to `path` and record it under that name."""
        path = Path(path)
        path.write_bytes(self.data)
        self.filename = path.name
        return self.record()


def prepare_export(
    results,
    *,
    actor: str,
    surface: str = audit.SURFACE_CLI,
    query: str | None = None,
    strategy: str | None = None,
    session_id: str | None = None,
    filename: str | None = None,
) -> Disclosure:
    """Build the workbook for `results` and the event that would account for it."""
    return Disclosure(
        data=export_vespa_xlsx(results).getvalue(),
        filename=filename or workbook_filename(),
        record_ids=audit.record_refs(results),
        actor=actor,
        surface=surface,
        query=query,
        strategy=strategy,
        session_id=session_id,
    )


def prepare_by_query_export(
    merged: list,
    per_query: list[tuple[str, list]],
    *,
    actor: str,
    surface: str = audit.SURFACE_CLI,
    query: str | None = None,
    strategy: str | None = None,
    filename: str | None = None,
) -> Disclosure:
    """Build the one-tab-per-query workbook and the event that would account for it."""
    # `merged` is the union of every per-query list, so it names every record
    # any tab holds.
    return Disclosure(
        data=export_by_query_xlsx(merged, per_query).getvalue(),
        filename=filename or workbook_filename(),
        record_ids=audit.record_refs(merged),
        actor=actor,
        surface=surface,
        query=query,
        strategy=strategy,
    )
