"""Replacing a DuckDB table with a fresh set of rows."""

from __future__ import annotations

import logging
from typing import Iterable

from .db_loader import quote_ident

log = logging.getLogger(__name__)

# The two columns every table leads with. `_record_id` is the upstream row id
# (or the importer's chosen id column, renamed); `_created_time` is when the
# tip was filed. Everything else follows alphabetically so two imports of the
# same dataset produce the same column order.
LEADING_COLUMNS = ("_record_id", "_created_time")


def ordered_columns(names: Iterable[str]) -> list[str]:
    """The column layout every table in this database uses."""
    rest = sorted(n for n in names if n not in LEADING_COLUMNS)
    return [*LEADING_COLUMNS, *rest]


def replace_table(conn, table_name: str, columns: list[str], rows: list[list]) -> None:
    """Replace `table_name` with `rows`, atomically."""
    col_defs = ", ".join(f"{quote_ident(c)} VARCHAR" for c in columns)
    placeholders = ", ".join(["?"] * len(columns))
    conn.execute("BEGIN TRANSACTION")
    try:
        conn.execute(f"DROP TABLE IF EXISTS {quote_ident(table_name)}")
        conn.execute(f"CREATE TABLE {quote_ident(table_name)} ({col_defs})")
        if rows:
            conn.executemany(
                f"INSERT INTO {quote_ident(table_name)} VALUES ({placeholders})", rows
            )
    except Exception:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")
    log.info(f"replaced table {table_name!r} ({len(rows)} rows, {len(columns)} cols)")
