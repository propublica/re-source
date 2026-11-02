#!/usr/bin/env python3
"""Mark documents already present in Vespa as indexed, so a feed skips them."""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.getLogger("httpr").setLevel(logging.WARNING)

from re_source.chunker import chunk_document
from re_source.config import VESPA_SYNC_STATE_PATH
from re_source.db_loader import get_all_documents
from re_source.sync_state import (
    filter_to_changed,
    load_sync_state,
    save_sync_state,
    update_state_for_docs,
)
from re_source.vespa_client import get_vespa_client

WORKERS = 16


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description="Reconcile sync_state against Vespa.")
    ap.add_argument("--write", action="store_true",
                    help="Actually update sync_state (backs it up first).")
    ap.add_argument("--workers", type=int, default=WORKERS)
    args = ap.parse_args()

    state = load_sync_state(VESPA_SYNC_STATE_PATH)
    docs = get_all_documents()
    todo = filter_to_changed(docs, state)
    print(f"{len(docs):,} documents, {len(state.indexed):,} tracked, "
          f"{len(todo):,} would be fed\n")
    if not todo:
        print("Nothing to reconcile.")
        return 0

    client = get_vespa_client()
    t0 = time.time()

    # Pass 1: one lookup per document. Most will be absent, and one miss is
    # enough to know the document needs feeding.
    plans = [(d, chunk_document(d)[0]) for d in todo]
    with_chunks = [(d, ch) for d, ch in plans if ch]
    no_chunks = len(plans) - len(with_chunks)

    def first_present(item):
        d, chunks = item
        return d.id if client.get("chunk", chunks[0].id) else None

    print(f"pass 1: checking {len(with_chunks):,} documents (first chunk only)...")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        candidate_ids = {r for r in ex.map(first_present, with_chunks) if r}
    print(f"        {len(candidate_ids):,} look present")

    # Pass 2: only for candidates, confirm EVERY chunk is there. A document
    # half-written must be re-fed, not skipped.
    candidates = [(d, ch) for d, ch in with_chunks if d.id in candidate_ids]

    def fully_present(item):
        d, chunks = item
        for ch in chunks:
            if not client.get("chunk", ch.id):
                return None
        return d

    print(f"pass 2: confirming all chunks for {len(candidates):,} documents...")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        complete = [d for d in ex.map(fully_present, candidates) if d is not None]

    partial = len(candidates) - len(complete)
    elapsed = time.time() - t0
    complete_ids = {d.id for d in complete}
    chunks_saved = sum(len(ch) for d, ch in candidates if d.id in complete_ids)

    print(f"\nscanned in {elapsed:.0f}s")
    print(f"  {len(complete):,} documents fully present  -> can be skipped")
    print(f"  {partial:,} partially present            -> will be re-fed (correct)")
    print(f"  {no_chunks:,} produce no chunks           -> nothing to check")
    print(f"\nskipping them avoids re-embedding ~{chunks_saved:,} chunks")
    if chunks_saved:
        print(f"at ~6 chunks/sec that is ~{chunks_saved / 6 / 3600:.1f} hours saved")

    if not args.write:
        print("\n(read-only; --write to apply)")
        print("\nSTOP THE FEED FIRST. A running feed rewrites sync_state at every")
        print("checkpoint, so it would overwrite whatever this writes.")
        return 0
    if not complete:
        print("\nNothing to mark.")
        return 0

    print("\nThis assumes the feed is STOPPED — a running feed rewrites sync_state")
    print("at every checkpoint and would overwrite this.")

    backup = VESPA_SYNC_STATE_PATH.with_suffix(".json.bak")
    if VESPA_SYNC_STATE_PATH.exists():
        shutil.copy2(VESPA_SYNC_STATE_PATH, backup)
        print(f"\nbacked up to {backup}")

    # Reuse the feeder's own updater, so the hash stored here is exactly what a
    # real feed would have stored — a later edit to the row is still detected as
    # changed by --incremental.
    update_state_for_docs(state, complete)
    save_sync_state(state, VESPA_SYNC_STATE_PATH)
    print(f"marked {len(complete):,} documents as indexed "
          f"({len(state.indexed):,} tracked)")
    print("\nRestart the feed with --incremental; it will skip them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
