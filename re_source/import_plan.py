"""What an upload would do, decided before anything is drawn."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd

from re_source import config
from re_source.datasets import is_externally_owned
from re_source.config import RECORD_ID_COLUMN
from re_source.db_loader import unsafe_record_id_chars
from re_source.importer import pending_hashes, refuse_destination, sanitize_table_name

logger = logging.getLogger(__name__)


class DatabaseView(Protocol):
    """The reads deciding an upload needs."""

    def list_tables(self) -> list[str]: ...
    # Every table in the database, not just the feedable ones. The plan needs
    # both: list_tables says what may be replaced, all_tables says what is
    # already there — and a name missing from the second is how an upload came
    # to silently overwrite a cache table.
    def all_tables(self) -> list[str]: ...
    def record_ids(self, table: str) -> set[str]: ...
    def count_rows(self, table: str) -> int: ...
    def content_hashes(self, table: str) -> dict[str, str]: ...
    def column_names(self, table: str) -> list[str]: ...



def suggest_table_name(filename: str) -> str:
    """The table name to prefill for an uploaded file — same rule the CLI uses
    for a path, so uploading `Housing 2026.csv` lands in `Housing_2026` either
    way."""
    return sanitize_table_name(Path(filename).stem)


def id_column_candidates(columns: list[str]) -> list[str]:
    """Columns to offer as the ID column, likeliest first."""
    preferred = ("record id", "_record_id", "id", "submission id", "response id")

    def rank(col: str) -> tuple[int, int]:
        lowered = col.strip().lower()
        for i, want in enumerate(preferred):
            if lowered == want:
                return (0, i)
        for i, want in enumerate(preferred):
            if want in lowered:
                return (1, i)
        return (2, 0)

    return sorted(columns, key=lambda c: (rank(c), columns.index(c)))


def uploadable_tables(tables: list[str]) -> list[str]:
    """The tables this page may show and replace — file imports only."""
    return [t for t in tables if not is_externally_owned(t)]


def preflight_warnings(
    df: pd.DataFrame,
    id_column: str,
    table_name: str,
    other_table_ids: dict[str, set[str]],
) -> list[tuple[str, str]]:
    """Things worth telling the uploader before they commit to the import."""
    out: list[tuple[str, str]] = []
    columns = list(df.columns)

    # Checked before anything else: a destination the import must refuse
    # outright. It also replaces the exists-and-confirm flow for these names —
    # they are deliberately absent from the uploadable table list, so without
    # this a synced table would look like a brand-new one and be replaced
    # with no confirmation at all.
    #
    # The rule is the importer's, asked here rather than restated: this page is
    # one of three doors and the other two never consulted it, so a rule owned
    # here was a rule only the page obeyed. What is left here is the ordering
    # (refused before anything below runs) and the level — the page shows the
    # same sentence the write would raise.
    refusal = refuse_destination(table_name)
    if refusal:
        out.append(("error", refusal))
        return out

    if id_column not in columns:
        out.append(("error", f"Column {id_column!r} is not in this file."))
        return out

    # The importer renames the chosen ID column to `_record_id`. A file that
    # already has that column ends up with two, and the import fails at CREATE.
    if "_record_id" in columns and id_column != "_record_id":
        out.append((
            "error",
            f"This file already has a `_record_id` column, which collides with "
            f"{id_column!r} once it is renamed. Remove or rename one of them.",
        ))

    # db_loader resolves the doc ID as `row.get("_record_id") or
    # row.get(RECORD_ID_COLUMN)`, and the importer renames the column picked
    # here to `_record_id` — so the choice made on this page wins and a column
    # that merely shares the name RECORD_ID_COLUMN is ordinary content.
    #
    # This was an error, on the opposite precedence: it told the uploader
    # "indexing reads 'id' first, so it would override your choice" and refused
    # the import. Both halves stopped being true when the precedence flipped,
    # and the rule went on rejecting uploads that would now behave correctly.
    # Kept as a warning because the surprise is real even though the outcome is
    # safe: a column named like the ID column is not the ID.
    if (
        RECORD_ID_COLUMN in columns
        and id_column != RECORD_ID_COLUMN
        and RECORD_ID_COLUMN != "_record_id"
    ):
        out.append((
            "warning",
            f"This file has a {RECORD_ID_COLUMN!r} column, but you picked "
            f"{id_column!r} as the ID. {id_column!r} is what will identify each "
            f"record; {RECORD_ID_COLUMN!r} will be indexed as ordinary content.",
        ))

    ids = df[id_column]
    blank = int((ids.isna() | (ids.astype(str).str.strip() == "")).sum())
    if blank:
        out.append(("warning", f"{blank:,} rows have a blank ID and will be skipped."))

    kept = ids[ids.notna() & (ids.astype(str).str.strip() != "")].astype(str)

    # Record ids end up inside quoted contexts — a Vespa document id, a
    # `parent_ref` in a document-select literal, a sync_state key. Those are
    # escaped where they are used, but an id containing a quote or a backslash
    # is never a real record id, so it is refused at the door rather than
    # trusted to every future call site.
    bad = {}
    for value in kept:
        chars = unsafe_record_id_chars(value)
        if chars:
            bad[value] = chars
    if bad:
        example, chars = sorted(bad.items())[0]
        out.append((
            "error",
            f"{len(bad):,} IDs contain characters that aren't allowed in a "
            f"record ID (e.g. {example!r}, which has {chars!r}). Record IDs "
            f"become document IDs and index references, so quotes, backslashes "
            f"and line breaks have to be removed before importing.",
        ))

    dupes = kept[kept.duplicated(keep=False)].unique()
    if len(dupes):
        examples = ", ".join(repr(d) for d in list(dupes)[:3])
        out.append((
            "error",
            f"{len(dupes):,} IDs appear more than once ({examples}…). "
            "Record IDs must be unique within a dataset.",
        ))

    if config.DOCUMENT_ID_SCHEME == "legacy":
        incoming = set(kept)
        for other, existing in sorted(other_table_ids.items()):
            if other == table_name:
                continue
            overlap = incoming & existing
            if overlap:
                out.append((
                    "error",
                    f"{len(overlap):,} of these IDs already exist in table {other!r} "
                    f"(e.g. {sorted(overlap)[0]!r}). Give this file unique IDs "
                    "before importing.",
                ))

    return out


def content_delta(old: dict[str, str], new: dict[str, str]) -> dict[str, int]:
    """Compare {doc_id: content_hash} before and after an import."""
    old_ids, new_ids = set(old), set(new)
    shared = old_ids & new_ids
    changed = sum(1 for doc_id in shared if old[doc_id] != new[doc_id])
    return {
        "unchanged": len(shared) - changed,
        "changed": changed,
        "deleted": len(old_ids - new_ids),
        "inserted": len(new_ids - old_ids),
    }


def column_delta(old_cols: set[str], new_cols: set[str]) -> dict[str, list[str]]:
    """Columns this upload adds and drops, ignoring `_`-prefixed internals —
    the importer synthesizes those, so they are never the uploader's doing."""
    def visible(cols):
        return {c for c in cols if not c.startswith("_")}

    old_v, new_v = visible(old_cols), visible(new_cols)
    return {"added": sorted(new_v - old_v), "removed": sorted(old_v - new_v)}


def describe_content_delta(delta: dict[str, int]) -> str:
    """The headline: what this import does to the table's rows."""
    if not any(delta.values()):
        return "No changes — this file matches the table exactly."

    line = (
        f"Deleting {delta['deleted']:,} rows, inserting {delta['inserted']:,} "
        f"new rows, updating {delta['changed']:,}."
    )
    if delta["unchanged"]:
        line += f" {delta['unchanged']:,} rows unchanged."
    return line


def _column_list(names: list[str], limit: int = 3) -> str:
    shown = ", ".join(names[:limit])
    return shown + ("…" if len(names) > limit else "")


def column_change_note(cols: dict[str, list[str]]) -> str | None:
    """Why every row can read as updated. None when the columns match."""
    added, removed = cols["added"], cols["removed"]
    if not added and not removed:
        return None

    bits = []
    if added:
        bits.append(f"adds {len(added):,} column(s) ({_column_list(added)})")
    if removed:
        bits.append(f"drops {len(removed):,} ({_column_list(removed)})")
    return (
        "This file " + " and ".join(bits) + ", so every row counts as updated "
        "even where the answers are unchanged."
    )


def replace_notice(table: str, existing_rows: int | None) -> str:
    """The plain statement of what is being replaced."""
    if existing_rows is None:
        return (
            f"Table `{table}` already exists, but its current row count could "
            f"not be read."
        )
    return f"Table `{table}` already exists and holds {existing_rows:,} rows."


def confirm_label(table: str, delta: dict[str, int] | None) -> str:
    """The checkbox label."""
    if delta is None:
        return f"Yes, replace the contents of `{table}`"
    return (
        f"Yes — delete {delta['deleted']:,}, insert {delta['inserted']:,} and "
        f"update {delta['changed']:,} rows in `{table}`"
    )


def _friendly_db_error(exc: Exception) -> str:
    """Turn a DuckDB failure into something an uploader can act on."""
    text = str(exc)
    if "lock" in text.lower():
        return (
            "The database is in use by another process — most likely an indexing "
            "or sync run. Wait for it to finish and try again."
        )
    return f"The import failed and nothing was changed: {text}"


@dataclass(frozen=True)
class ImportPlan:
    """What importing this frame into this table would do."""

    table: str = ""
    problems: tuple[tuple[str, str], ...] = ()
    is_replace: bool = False
    existing_rows: int | None = None
    content: dict | None = None
    columns: dict | None = None
    comparable: bool = True

    @property
    def blocked(self) -> bool:
        """Any error at all stops the import. Warnings do not."""
        return any(level == "error" for level, _ in self.problems)

    @property
    def needs_confirmation(self) -> bool:
        """Replacing a table destroys what is there; creating one doesn't."""
        return self.is_replace and not self.blocked

    def may_proceed(self, confirmed: bool) -> bool:
        """Whether to run the import."""
        if self.blocked:
            return False
        return confirmed if self.needs_confirmation else True

    @property
    def notice(self) -> str | None:
        """What the page says above the confirmation, or nothing to say."""
        return replace_notice(self.table, self.existing_rows) if self.is_replace else None

    @property
    def change_summary(self) -> str | None:
        if not self.is_replace:
            return None
        if not self.comparable:
            return ("Could not compare this file against the current rows, so "
                    "the exact number of changes is unknown.")
        return describe_content_delta(self.content)

    @property
    def column_note(self) -> str | None:
        return column_change_note(self.columns) if self.columns else None

    @property
    def confirmation(self) -> str:
        """The wording on the checkbox that has to be ticked."""
        return confirm_label(self.table, self.content)


def plan(df: pd.DataFrame, *, id_column: str, table: str,
         db: DatabaseView) -> ImportPlan:
    """What importing `df` into `table` would do."""
    tables = db.list_tables()

    other_ids = {}
    if config.DOCUMENT_ID_SCHEME == "legacy":
        other_ids = {t: db.record_ids(t) for t in db.all_tables() if t != table}
    problems = tuple(preflight_warnings(df, id_column, table, other_ids))
    if any(level == "error" for level, _ in problems):
        # Nothing below is meaningful for a destination we are refusing, and
        # reading the target table would cost a scan for an answer nobody sees.
        return ImportPlan(table=table, problems=problems)

    # Only file imports are replaceable; a synced table name never reaches
    # here, because preflight_warnings rejected it above.
    if table not in uploadable_tables(tables):
        return ImportPlan(table=table, problems=problems)

    try:
        existing_rows = db.count_rows(table)
    except Exception:
        logger.warning(f"could not count rows in {table!r}", exc_info=True)
        existing_rows = None

    try:
        content = content_delta(db.content_hashes(table),
                                pending_hashes(df, id_column, table))
        columns = column_delta(set(db.column_names(table)), set(df.columns))
        comparable = True
    except Exception as exc:
        # A preview is a nicety; never let it block the import path.
        logger.warning(f"could not compare {table!r} against the upload: {exc}")
        content, columns, comparable = None, None, False

    return ImportPlan(
        table=table,
        problems=problems,
        is_replace=True,
        existing_rows=existing_rows,
        content=content,
        columns=columns,
        comparable=comparable,
    )
