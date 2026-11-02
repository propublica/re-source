"""Load imported records from DuckDB."""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import duckdb

from .config import DOCUMENT_ID_SCHEME, DUCKDB_PATH, RECORD_ID_COLUMN
from .datasets import DATASET_CATALOG_TABLE

logger = logging.getLogger(__name__)

# Increment when the Vespa document derived from an unchanged row changes. This
# forces the next incremental feed to refresh existing documents instead of
# leaving them in the previous index shape.
INDEX_FORMAT_VERSION = 2

# Tables in the DuckDB that are NOT source data and must never be loaded as
# documents. Without this, get_all_documents would treat their rows as tips —
# enriching/indexing them and creating a feedback loop.
#
# Every table the project's own scripts write into this database belongs here.
# `field_inventory` and `question_templates` were missing, and field_inventory
# is the costly omission: it can store verbatim sample values per column, so
# indexing it would surface copied submissions with no source provenance on the
# card. A deployment policy could not catch that reliably because it sees the
# container column (`sample_values`), not the original field whose values were
# copied inside it.
#
# scripts/feed_progress.py used to re-add these two names itself, which is the
# evidence this set was incomplete rather than deliberately narrow.
NON_SOURCE_TABLES = {
    DATASET_CATALOG_TABLE,
    "enrichment",
    "field_inventory",
    "question_templates",
}


# A record id becomes a Vespa document id, a `parent_ref` inside a
# document-select string literal, and a key in sync_state. Every one of those
# contexts is escaped at the point of use, but an id carrying these characters
# has no legitimate reading and is far more likely to be a mistake (or an
# attempt) than a real imported record id, so the upload gate refuses it rather
# than relying on every future call site to remember to escape.
_UNSAFE_ID_CHARS = '"\\\n\r'


def unsafe_record_id_chars(value: str) -> str:
    """The characters in `value` that make it unsafe as a record id, or ""."""
    return "".join(sorted({c for c in str(value) if c in _UNSAFE_ID_CHARS}))


def quote_ident(name: str) -> str:
    """Quote a table or column name for DuckDB, escaping internal double quotes."""
    return '"' + name.replace('"', '""') + '"'


def make_document_id(source_file: str, record_id: str) -> str:
    """Return the Vespa ID for a dataset record."""
    raw = str(record_id)
    if DOCUMENT_ID_SCHEME == "legacy":
        return raw
    value = json.dumps([source_file, raw], ensure_ascii=False, separators=(",", ":"))
    return "ds1_" + hashlib.sha256(value.encode()).hexdigest()


@dataclass
class Document:
    """A document loaded from DuckDB."""

    id: str  # Unique document ID (table_recordid)
    source_file: str  # Table name
    fields: dict[str, any]  # All row fields
    content_hash: str  # Hash for change detection

    # For compatibility with existing code
    text: str = ""  # Will be set from fields


def compute_content_hash(row_dict: dict) -> str:
    """Compute a hash of the row content for change detection.

    Args:
        row_dict: The row data as a dictionary

    Returns:
        16-character hash string
    """
    # Sort keys for consistent hashing
    content = json.dumps(
        {"index_format": INDEX_FORMAT_VERSION, "fields": row_dict},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def extract_text(fields: dict[str, any]) -> str:
    """Concatenate all fields into searchable text.

    Args:
        fields: Dictionary of field name -> value

    Returns:
        Concatenated text with field names as prefixes
    """
    text_parts = []
    for field_name, value in fields.items():
        # Skip internal/metadata fields
        if field_name.startswith("_"):
            continue
        if value and str(value).strip():
            text_parts.append(f"{field_name}: {value}")
    return "\n".join(text_parts)


def document_from_row(row_dict: dict, source_file: str) -> Optional[Document]:
    """Build a Document from one DuckDB row, or None if it has no searchable text."""
    # `_record_id` first: both writers put the row's real identity there.
    # RECORD_ID_COLUMN (default "id") used to win, so a column merely *named*
    # `id` outranked it and a value a tip submitter typed became the document
    # id — two people answering alike collided and one was silently lost.
    # Kept as a fallback for a table written by neither importer.
    doc_id = row_dict.get("_record_id") or row_dict.get(RECORD_ID_COLUMN)
    if not doc_id:
        # Python's built-in `hash()` of strings is randomized per
        # process (PYTHONHASHSEED), so it can't be used here — every
        # run would assign different IDs to the same row, breaking
        # incremental sync's content-hash comparison. Use SHA-256.
        content = json.dumps(row_dict, sort_keys=True, default=str)
        doc_id = hashlib.sha256(content.encode()).hexdigest()[:16]
    doc_id = make_document_id(source_file, str(doc_id))
    text = extract_text(row_dict)

    if not text.strip():
        return None

    return Document(
        id=doc_id,
        source_file=source_file,
        fields=row_dict,
        content_hash=compute_content_hash(row_dict),
        text=text,
    )


def load_table(conn: duckdb.DuckDBPyConnection, table_name: str) -> Iterator[Document]:
    """Load all rows from a single DuckDB table."""
    rows = conn.execute(f"SELECT * FROM {quote_ident(table_name)}").fetchall()
    columns = [desc[0] for desc in conn.description]

    for row in rows:
        doc = document_from_row(dict(zip(columns, row)), table_name)
        if doc is not None:
            yield doc


def _connect_read_only() -> duckdb.DuckDBPyConnection:
    """Read-only connection to the project DuckDB, or FileNotFoundError."""
    if not DUCKDB_PATH.exists():
        raise FileNotFoundError(
            f"DuckDB database not found: {DUCKDB_PATH}\n"
            "Run 'python scripts/import_file.py <path>' to import a dataset."
        )
    return duckdb.connect(str(DUCKDB_PATH), read_only=True)


def _source_tables(conn: duckdb.DuckDBPyConnection) -> list[str]:
    """Source-data table names on an open connection, minus the ones that are
    not tips (see NON_SOURCE_TABLES)."""
    names = [row[0] for row in conn.execute("SHOW TABLES").fetchall()]
    return [n for n in names if n not in NON_SOURCE_TABLES]


def list_tables() -> list[str]:
    """Source-data table names in the DuckDB, sorted."""
    if not DUCKDB_PATH.exists():
        return []
    conn = _connect_read_only()
    try:
        return sorted(_source_tables(conn))
    finally:
        conn.close()


def all_tables() -> list[str]:
    """Every table in the DuckDB, sorted — including the non-source ones."""
    if not DUCKDB_PATH.exists():
        return []
    conn = _connect_read_only()
    try:
        return sorted(row[0] for row in conn.execute("SHOW TABLES").fetchall())
    finally:
        conn.close()


def count_rows(table: str) -> int:
    """Row count for one table. Raises duckdb.Error if it doesn't exist."""
    conn = _connect_read_only()
    try:
        return conn.execute(f"SELECT count(*) FROM {quote_ident(table)}").fetchone()[0]
    finally:
        conn.close()


def column_names(table: str) -> list[str]:
    """Column names of one table, in declaration order; [] if it doesn't exist."""
    conn = _connect_read_only()
    try:
        return _column_names(conn, table)
    finally:
        conn.close()


def _column_names(conn: duckdb.DuckDBPyConnection, table: str) -> list[str]:
    """Column names on an open connection. Empty list for an unknown table."""
    rows = conn.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = ? ORDER BY ordinal_position",
        [table],
    ).fetchall()
    return [row[0] for row in rows]


def content_hashes(table: str) -> dict[str, str]:
    """{doc_id: content_hash} for one table, as the feed pipeline sees it."""
    conn = _connect_read_only()
    try:
        return {doc.id: doc.content_hash for doc in load_table(conn, table)}
    finally:
        conn.close()


def record_ids(table: str) -> set[str]:
    """The `_record_id` values in one table."""
    conn = _connect_read_only()
    try:
        if "_record_id" not in set(_column_names(conn, table)):
            return set()
        rows = conn.execute(
            f"SELECT DISTINCT _record_id FROM {quote_ident(table)}"
        ).fetchall()
        return {str(r[0]) for r in rows if r[0] is not None}
    finally:
        conn.close()


def load_all_tables() -> Iterator[Document]:
    """Load all rows from all tables in DuckDB."""
    if not DUCKDB_PATH.exists():
        raise FileNotFoundError(
            f"DuckDB database not found: {DUCKDB_PATH}\n"
            "Run 'python scripts/import_file.py <path>' to import a dataset."
        )

    conn = duckdb.connect(str(DUCKDB_PATH), read_only=True)

    try:
        tables = _source_tables(conn)

        if not tables:
            raise ValueError(
                f"No tables found in {DUCKDB_PATH}\n"
                "Run 'python scripts/import_file.py <path>' to import a dataset."
            )

        logger.info(f"Found {len(tables)} tables in DuckDB")

        total_docs = 0
        for table_name in tables:
            logger.info(f"Loading: {table_name}")
            doc_count = 0

            for doc in load_table(conn, table_name):
                doc_count += 1
                yield doc

            total_docs += doc_count
            logger.info(f"  Loaded {doc_count} documents")

        logger.info(f"Total: {total_docs} documents")

    finally:
        conn.close()


def get_document_count() -> int:
    """Count total documents across all tables in DuckDB."""
    return sum(1 for _ in load_all_tables())


def get_all_documents() -> list[Document]:
    """Load all documents into a list (for incremental indexing)."""
    return list(load_all_tables())
