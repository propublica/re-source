#!/usr/bin/env python3
"""How far along is the feed, and how long is left."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.getLogger("httpr").setLevel(logging.WARNING)

import duckdb

from re_source.config import DUCKDB_PATH, VESPA_SYNC_STATE_PATH
from re_source.db_loader import NON_SOURCE_TABLES, quote_ident
from re_source.sync_state import load_sync_state
from re_source.vespa_client import get_vespa_client


def total_rows() -> int:
    """Cheap COUNT(*) per source table — an UPPER BOUND on documents, not the
    document count.
    """
    conn = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    try:
        total = 0
        for (table,) in conn.execute("SHOW TABLES").fetchall():
            if table in NON_SOURCE_TABLES:
                continue
            # quote_ident, not a hand-written pair of quotes: the admin page's
            # table-name field is free text and never passes through
            # sanitize_table_name (only its prefill does), so a name carrying a
            # double quote imports fine and then closes the identifier here.
            total += conn.execute(
                f"SELECT count(*) FROM {quote_ident(table)}"
            ).fetchone()[0]
        return total
    finally:
        conn.close()


def chunk_count(client) -> int:
    r = client.query(yql="select * from chunk where true", ranking="unranked", hits=0)
    return r["root"]["fields"]["totalCount"]


def main() -> int:
    ap = argparse.ArgumentParser(description="Feed progress. Read-only.")
    ap.add_argument("--seconds", type=int, default=30,
                    help="Sampling window for the rate (default 30).")
    ap.add_argument("--exact", action="store_true",
                    help="Count documents properly instead of rows. Accurate, but "
                         "loads the whole corpus.")
    args = ap.parse_args()

    done = len(load_sync_state(VESPA_SYNC_STATE_PATH).indexed)
    if args.exact:
        from re_source.db_loader import get_document_count
        total = get_document_count()
        label = "documents"
    else:
        total = total_rows()
        label = "rows (upper bound; --exact for the true document count)"
    pct = 100.0 * done / total if total else 0.0
    print(f"documents: {done:,} / {total:,} {label}  ({pct:.1f}%)")
    if not args.exact:
        print("           rows with no content never become documents, so real")
        print("           progress is HIGHER than this.")

    client = get_vespa_client()
    a = chunk_count(client)
    print(f"chunks in Vespa: {a:,}   (sampling {args.seconds}s...)")
    time.sleep(args.seconds)
    b = chunk_count(client)
    # Sample documents too. Chunks alone cannot tell a stalled feed from one
    # working through a table whose columns are all excluded — those documents
    # checkpoint normally and produce nothing.
    done_after = len(load_sync_state(VESPA_SYNC_STATE_PATH).indexed)

    rate = (b - a) / args.seconds
    doc_rate = (done_after - done) / args.seconds
    print(f"chunks now:      {b:,}   ({b - a:+,} in {args.seconds}s = {rate:.1f}/sec)")
    print(f"documents now:   {done_after:,}   ({done_after - done:+,})")

    if rate <= 0 and doc_rate > 0:
        print("\nDocuments are advancing but no chunks are being produced — the feed")
        print("is working through a table whose columns are all excluded. Normal;")
        print("the chunk count picks up at the next table with content.")
        return 0
    if rate <= 0:
        print("\nNeither documents nor chunks are advancing. Either the feed is not")
        print("running, or it is mid-group — checkpoints land every 100 documents,")
        print("so re-run in a minute before concluding it stalled.")
        return 0

    remaining_docs = total - done
    if remaining_docs > 0 and done:
        chunks_per_doc = b / done
        eta_s = (remaining_docs * chunks_per_doc) / rate
        print(f"\n~{chunks_per_doc:.1f} chunks/doc so far, {remaining_docs:,} docs left")
        print(f"ETA at this rate: {eta_s / 3600:.1f} hours")
        print("\nchunks/doc is a running average and drifts upward: documents are fed")
        print("table by table, and forms differ hugely — a short intake form makes a")
        print("few chunks per row, a long narrative form makes dozens. Early ETAs are")
        print("therefore optimistic; trust the number more as it converges.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
