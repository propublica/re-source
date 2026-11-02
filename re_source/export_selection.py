"""Which results are ticked for the Excel export."""

from __future__ import annotations

from typing import MutableMapping

EXPORT_PREFIX = "export::"


def identity(r) -> str | None:
    """Stable per-record identity: `{source_file}::{record or doc id}`, or None."""
    rec = (getattr(r, "fields", None) or {}).get("_record_id")
    ident = rec or getattr(r, "doc_id", None)
    if not ident:
        return None
    return f"{getattr(r, 'source_file', '')}::{ident}"


class ExportSelection:
    """The include-in-export choice for each result, over one state mapping."""

    def __init__(self, state: MutableMapping) -> None:
        self._state = state

    @staticmethod
    def key(r) -> str | None:
        """This row's include-in-export key, or None if it cannot be named."""
        ident = identity(r)
        return f"{EXPORT_PREFIX}{ident}" if ident else None

    def includes(self, r) -> bool:
        """True unless this row has been explicitly dropped.

        Returns:
            True when the row is bound for the workbook. Defaults to True.
        """
        key = self.key(r)
        if key is None:
            return False
        return self._state.get(key, True)

    def set(self, r, included: bool) -> None:
        """Record a choice. A row with no identity has no choice to record."""
        key = self.key(r)
        if key is None:
            return
        self._state[key] = bool(included)

    def toggle(self, r) -> None:
        """Flip one row's choice. The on_click/on_change for both views."""
        self.set(r, not self.includes(r))

    def rows(self, results) -> list:
        """The subset of `results` bound for the workbook, in order."""
        return [r for r in results if self.includes(r)]

    def clear(self) -> None:
        """Forget every choice."""
        for key in [k for k in self._state if str(k).startswith(EXPORT_PREFIX)]:
            self._state.pop(key, None)
