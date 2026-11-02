"""Import a CSV or XLSX file into DuckDB as a table matching the shape
the rest of the pipeline expects.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import duckdb
import pandas as pd

from .config import DUCKDB_PATH
from .datasets import is_externally_owned
from .db_loader import NON_SOURCE_TABLES, document_from_row
from .duckdb_table import ordered_columns, replace_table

log = logging.getLogger(__name__)


def sanitize_table_name(stem: str) -> str:
    """Make a filename stem a valid DuckDB identifier: word chars only,
    no leading digit."""
    safe = re.sub(r"\W+", "_", stem).strip("_") or "table"
    if safe[0].isdigit():
        safe = "t_" + safe
    return safe


def read_file(path: Path, sheet: str | None) -> pd.DataFrame:
    """Read CSV or XLSX into a DataFrame.

    Returns:
        The parsed frame, every cell cast to str; empty cells stay NaN.
    """
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
    elif path.suffix.lower() in (".xlsx", ".xls"):
        # If sheet is None, pandas reads the first sheet.
        df = pd.read_excel(path, sheet_name=sheet or 0, dtype=str)
    else:
        raise ValueError(f"Unsupported extension: {path.suffix!r} (want .csv, .xlsx, or .xls)")
    return df


def normalize_for_import(
    df: pd.DataFrame,
    id_column: str,
) -> tuple[list[list], list[str]]:
    """Shape a parsed frame into the (rows, columns) actually written to DuckDB."""
    # Rename the chosen id column to `_record_id` so db_loader sees it;
    # a sync writes the same column for the rows it owns.
    df = df.rename(columns={id_column: "_record_id"})

    if "_created_time" not in df.columns:
        df["_created_time"] = ""
    df = df[ordered_columns(df.columns)]

    # Normalize values to str (DuckDB VARCHAR), preserving None for missing.
    def _norm(v):
        if v is None or pd.isna(v):
            return None
        return str(v)

    rows = [[_norm(v) for v in row] for row in df.itertuples(index=False, name=None)]
    return rows, list(df.columns)


def pending_hashes(
    df: pd.DataFrame,
    id_column: str,
    table_name: str,
) -> dict[str, str]:
    """{doc_id: content_hash} for the rows importing `df` *would* write."""
    rows, cols = normalize_for_import(df, id_column)
    hashes: dict[str, str] = {}
    for row in rows:
        doc = document_from_row(dict(zip(cols, row)), table_name)
        if doc is not None:
            hashes[doc.id] = doc.content_hash
    return hashes


def refuse_destination(table_name: str) -> str | None:
    """Why a file import must not write to `table_name`, or None if it may.

    The two destinations a file import may never replace, and the reason in
    the words the uploader should see. Returned rather than raised so the
    upload page can show it as an error before anything happens, while
    `import_dataframe` raises the same sentence if a caller gets that far.

    This lives beside the write rather than in the upload page's rules because
    the page is not the only door. `scripts/import_file.py` calls the importer
    directly and never touched those rules, so
    `--table-name enrichment` dropped the author-profile cache, and naming a
    synced table replaced it wholesale. A rule that only one of three callers
    consults is a convention, not a guard.

    A sync is deliberately unaffected: it calls replace_table itself,
    because writing the table it owns is exactly its job.
    """
    if is_externally_owned(table_name):
        return (
            f"`{table_name}` is kept up to date by a sync, not by upload. "
            f"Writing into it would destroy the synced data and be undone by "
            f"the next sync. Pick a different table name."
        )
    if table_name in NON_SOURCE_TABLES:
        return (
            f"`{table_name}` is written and owned by this project's own "
            f"scripts, not uploaded. Replacing it would destroy what they have "
            f"built up. Pick a different table name."
        )
    return None


def import_dataframe(
    df: pd.DataFrame,
    id_column: str,
    table_name: str,
) -> dict:
    """Write an already-parsed DataFrame into DuckDB as `table_name`."""
    refusal = refuse_destination(table_name)
    if refusal:
        raise ValueError(refusal)

    if id_column not in df.columns:
        raise ValueError(
            f"id column {id_column!r} not found. Available columns: {list(df.columns)}"
        )

    # Drop rows with empty ID
    before = len(df)
    df = df[df[id_column].notna() & (df[id_column].astype(str).str.strip() != "")]
    dropped_blank_ids = before - len(df)
    if dropped_blank_ids:
        log.warning(f"dropped {dropped_blank_ids} rows with empty {id_column!r}")

    dupes = df[df[id_column].duplicated(keep=False)][id_column].unique()
    if len(dupes) > 0:
        raise ValueError(
            f"{len(dupes)} duplicate record IDs: {list(dupes)[:5]}"
        )

    rows, cols = normalize_for_import(df, id_column)

    log.info(f"writing to DuckDB table {table_name!r} ({len(rows)} rows, {len(cols)} cols)")

    # A fresh deployment may have no data directory yet — the upload page can
    # be the very first thing that writes the database. A sync does the
    # same before its first connect.
    DUCKDB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = duckdb.connect(str(DUCKDB_PATH))
    try:
        replace_table(conn, table_name, cols, rows)
    finally:
        conn.close()

    return {
        "table": table_name,
        "rows": len(rows),
        "cols": len(cols),
        "dropped_blank_ids": dropped_blank_ids,
        "duplicate_ids": 0,
    }


def import_file(
    path: Path,
    id_column: str,
    table_name: str | None,
    sheet: str | None,
) -> dict:
    """Parse a CSV/XLSX file and import it.

    Args:
        table_name: destination table; defaults to a sanitized filename stem.
    """
    df = read_file(path, sheet)
    log.info(f"read {len(df)} rows × {len(df.columns)} columns from {path}")

    table = table_name or sanitize_table_name(path.stem)
    stats = import_dataframe(df, id_column=id_column, table_name=table)
    return {**stats, "source": str(path)}
