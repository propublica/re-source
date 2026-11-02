"""Vespa-side feed pipeline."""

from __future__ import annotations

import argparse
import logging
import time
from typing import Iterable

from .chunker import AtomicChunk, ParentRow, chunk_documents
from .config import VESPA_SYNC_STATE_PATH
from .db_loader import Document, get_all_documents
from .sync_state import (
    SyncState,
    filter_to_changed,
    get_deleted_doc_ids,
    load_sync_state,
    remove_from_state,
    save_sync_state,
    update_state_for_docs,
)
from .vespa_chunks import quote_yql_string
from .vespa_client import VespaClient

logger = logging.getLogger(__name__)


def keyword_text(fields: dict) -> str:
    """Every non-blank stored field, for lexical search without embeddings."""
    return "\n".join(
        f"{name}: {value}"
        for name, value in fields.items()
        if value is not None and str(value).strip()
    )


def parent_doc(parent: ParentRow) -> tuple[str, dict]:
    return (
        parent.id,
        {
            "parent_id": parent.id,
            "source_file": parent.source_file,
            "full_row": parent.fields,
            "keyword_text": keyword_text(parent.fields),
        },
    )


def chunk_doc(chunk: AtomicChunk) -> tuple[str, dict]:
    """Convert an AtomicChunk to a feed tuple."""
    return (
        chunk.id,
        {
            "chunk_id": chunk.id,
            "parent_id": chunk.parent_id,
            "parent_ref": f"id:tips:parent::{chunk.parent_id}",
            "column_name": chunk.column_name,
            "column_key": chunk.column_key,
            "source_file": chunk.source_file,
            "text": chunk.text,
        },
    )


def _profiles_for(docs: list[Document]) -> dict[str, str]:
    """Saved, current author profiles for the rows about to be fed."""
    # Lazy: enrich imports this module's feed helpers.
    from .enrich import current_profiles

    return current_profiles(docs)


def _profile_chunks(parents: list[ParentRow], profiles: dict[str, str]) -> list[AtomicChunk]:
    """author_profile chunks to feed alongside re-fed parents."""
    from .enrich import profile_chunk

    # Re-feeding a parent deletes all its chunks, profile included, and the
    # enrichment feed's own state still records the profile as fed, so it would
    # never come back. Carry it here instead. The parent's full_row is left as
    # chunked, keeping the AI-written profile out of keyword_text.
    return [
        profile_chunk(p.id, p.source_file, profiles[p.id])
        for p in parents
        if p.id in profiles
    ]


# ----- delete helpers --------------------------------------------------


def cascade_delete_parents(client: VespaClient, parent_ids: Iterable[str]) -> list[str]:
    """Delete each parent doc and every chunk referencing it.

    Returns:
        The ids actually deleted. The caller prunes sync_state by them, so a
        failure must not read as a success.
    """
    deleted: list[str] = []
    for pid in parent_ids:
        selection = f'chunk.parent_ref=="id:tips:parent::{quote_yql_string(pid)}"'
        try:
            client.delete_by_selection("chunk", selection)
            client.delete("parent", pid)
            deleted.append(pid)
        except Exception as e:
            logger.warning(f"  failed to delete {pid}: {e}")
    return deleted


def delete_source_file(client: VespaClient, source_file: str) -> None:
    """Delete every indexed document belonging to one source file."""
    value = quote_yql_string(source_file)
    client.delete_by_selection("chunk", f'chunk.source_file=="{value}"')
    client.delete_by_selection("parent", f'parent.source_file=="{value}"')


# ----- main pipeline ---------------------------------------------------


def feed_incremental(
    client: VespaClient,
    full: bool = False,
    checkpoint_every: int = 100,
) -> dict:
    """Sweep DuckDB, sync to Vespa, checkpoint progress every N docs.

    Args:
        full: if True, wipe the Vespa schemas first and reset sync_state.
        checkpoint_every: docs per checkpoint group. Smaller = faster
            recovery from a crash. Default 100.
    """
    t0 = time.time()
    state = load_sync_state(VESPA_SYNC_STATE_PATH)

    logger.info("loading documents from DuckDB")
    all_docs: list[Document] = get_all_documents()
    logger.info(f"DuckDB has {len(all_docs)} documents")

    if full:
        logger.info("--full: wiping chunk + parent schemas in Vespa")
        # Verified: a wipe that quietly leaves documents behind is worse than
        # one that fails, because the survivors belong to the OLD configuration
        # and nothing will overwrite them.
        client.delete_all_verified("chunk")
        client.delete_all_verified("parent")
        state = SyncState()
        save_sync_state(state, VESPA_SYNC_STATE_PATH)
        docs_to_index = all_docs
    else:
        deleted_ids = get_deleted_doc_ids(all_docs, state)
        if deleted_ids:
            logger.info(f"pruning {len(deleted_ids)} deleted parents from Vespa")
            # Only forget what Vespa confirmed gone. Deletions are derived as
            # state-minus-DuckDB, so an id dropped from state can never appear
            # in deleted_ids again — forgetting one whose delete failed leaves
            # the document in the index with nothing left that knows to remove
            # it. These are rows deleted at the source, which for this corpus
            # includes a tip withdrawn by its source or removed on legal
            # advice, so "still searchable and now unreachable by any prune"
            # is the worst outcome available here. Keeping the failures in
            # state costs one retry on the next run.
            deleted_ok = cascade_delete_parents(client, deleted_ids)
            failed = [pid for pid in deleted_ids if pid not in set(deleted_ok)]
            if failed:
                logger.error(
                    f"{len(failed)} parents could not be deleted from Vespa and are "
                    f"STILL INDEXED; kept in sync_state so the next run retries them: "
                    f"{failed[:5]}{'...' if len(failed) > 5 else ''}"
                )
            remove_from_state(state, deleted_ok)
            save_sync_state(state, VESPA_SYNC_STATE_PATH)

        docs_to_index = filter_to_changed(all_docs, state)
        logger.info(f"previously indexed: {len(state.indexed)}")
        logger.info(f"new or changed: {len(docs_to_index)}")

    if not docs_to_index:
        logger.info("nothing to index — Vespa is up to date")
        return {
            "mode": "full" if full else "incremental",
            "parents": 0,
            "chunks": 0,
            "checkpoints_saved": 0,
            "checkpoints_skipped": 0,
            "delete_failures": 0,
            "elapsed_seconds": round(time.time() - t0, 1),
        }

    profiles = _profiles_for(docs_to_index)
    total_parents = 0
    total_chunks = 0
    saved_checkpoints = 0
    skipped_checkpoints = 0
    delete_failures = 0
    total_groups = (len(docs_to_index) + checkpoint_every - 1) // checkpoint_every

    for i in range(0, len(docs_to_index), checkpoint_every):
        group = docs_to_index[i : i + checkpoint_every]
        group_num = i // checkpoint_every + 1
        logger.info(f"--- checkpoint {group_num}/{total_groups}: {len(group)} docs ---")

        # Cascade-delete stale chunks for parents already in state (i.e.
        # changed, not new). Done per-group rather than upfront so the
        # window where a parent has empty chunks if we crash is one group
        # wide, not the whole corpus.
        changed_in_group = [d.id for d in group if d.id in state.indexed]
        ready_group = group
        if changed_in_group:
            logger.info(f"  clearing stale chunks for {len(changed_in_group)} changed parents")
            deleted = set(cascade_delete_parents(client, changed_in_group))
            failed_deletes = set(changed_in_group) - deleted
            delete_failures += len(failed_deletes)
            ready_group = [d for d in group if d.id not in failed_deletes]

        if not ready_group:
            skipped_checkpoints += 1
            continue

        parents: list[ParentRow] = []
        chunks: list[AtomicChunk] = []
        for chunk_list, parent in chunk_documents(ready_group):
            parents.append(parent)
            chunks.extend(chunk_list)
        chunks.extend(_profile_chunks(parents, profiles))

        p_res = client.feed_batch("parent", (parent_doc(p) for p in parents))
        c_res = client.feed_batch("chunk", (chunk_doc(c) for c in chunks))
        logger.info(
            f"  fed: parents={p_res.successes}/{p_res.successes + p_res.failures}, "
            f"chunks={c_res.successes}/{c_res.successes + c_res.failures}"
        )
        total_parents += p_res.successes
        total_chunks += c_res.successes

        # Conservative checkpoint: only mark the group as indexed if every
        # feed in the group succeeded. Any failure → skip checkpoint, log,
        # and let the next --incremental run retry the whole group.
        if p_res.failures == 0 and c_res.failures == 0:
            update_state_for_docs(state, ready_group)
            save_sync_state(state, VESPA_SYNC_STATE_PATH)
            saved_checkpoints += 1
            logger.info(f"  checkpoint saved ({len(state.indexed)} total tracked)")
        else:
            skipped_checkpoints += 1
            logger.warning(
                f"  group {group_num} had failures (parent={p_res.failures}, "
                f"chunk={c_res.failures}); not checkpointing. Re-run to retry."
            )

    return {
        "mode": "full" if full else "incremental",
        "parents": total_parents,
        "chunks": total_chunks,
        "checkpoints_saved": saved_checkpoints,
        "checkpoints_skipped": skipped_checkpoints,
        "delete_failures": delete_failures,
        "elapsed_seconds": round(time.time() - t0, 1),
    }


def feed_source_file(
    source_file: str,
    client: VespaClient,
    record_state: bool = False,
) -> dict:
    """Upsert one table only."""
    t0 = time.time()
    logger.info(f"upserting source_file={source_file!r} only")

    documents = [d for d in get_all_documents() if d.source_file == source_file]
    if not documents:
        raise RuntimeError(f"No documents found for source_file={source_file!r}")
    logger.info(f"matched {len(documents)} parents")

    parents: list[ParentRow] = []
    chunks: list[AtomicChunk] = []
    for chunk_list, parent in chunk_documents(documents):
        parents.append(parent)
        chunks.extend(chunk_list)
    chunks.extend(_profile_chunks(parents, _profiles_for(documents)))
    logger.info(f"chunked: {len(parents)} parents, {len(chunks)} chunks")

    state = None
    if record_state:
        state = load_sync_state(VESPA_SYNC_STATE_PATH)
        remove_from_state(state, {document.id for document in documents})
        save_sync_state(state, VESPA_SYNC_STATE_PATH)

    delete_source_file(client, source_file)

    p_res = client.feed_batch("parent", (parent_doc(p) for p in parents))
    c_res = client.feed_batch("chunk", (chunk_doc(c) for c in chunks))

    recorded = False
    if record_state:
        if p_res.failures == 0 and c_res.failures == 0:
            update_state_for_docs(state, documents)
            save_sync_state(state, VESPA_SYNC_STATE_PATH)
            recorded = True
            logger.info(f"recorded {len(documents)} docs in sync_state")
        else:
            logger.warning(
                f"not recording sync_state: parent={p_res.failures} chunk="
                f"{c_res.failures} feeds failed; re-run to retry"
            )

    return {
        "mode": "source_file",
        "source_file": source_file,
        "parents": p_res.successes,
        "chunks": c_res.successes,
        "parent_failures": p_res.failures,
        "chunk_failures": c_res.failures,
        "state_recorded": recorded,
        "elapsed_seconds": round(time.time() - t0, 1),
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Feed chunks + parents into Vespa.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--incremental", action="store_true",
                       help="(default) Sync changed + deleted docs against sync_state.")
    group.add_argument("--full", action="store_true",
                       help="Wipe schemas and re-feed everything from scratch.")
    group.add_argument("--source-file", help="Upsert one DuckDB table only; skip sync_state.")
    parser.add_argument("--checkpoint-every", type=int, default=100,
                        help="Save sync_state every N documents in incremental/full "
                             "mode. A crash loses at most this many docs' work "
                             "(re-run resumes). Default 100.")
    args = parser.parse_args()

    # Classification and an optional deployment policy are cached at process
    # start, so surface what this feed will use before it does any work.
    from . import config
    from .chunker import load_field_classes

    classes = load_field_classes()
    policy = config.CHUNK_POLICY_MODULE.strip() or "public default (no exclusions)"
    logger.info("config: chunk policy=%s, %d classified columns", policy, len(classes))

    client = VespaClient()
    logger.info(f"feeding to {client.endpoint}")

    if args.source_file:
        stats = feed_source_file(args.source_file, client)
    else:
        stats = feed_incremental(
            client,
            full=args.full,
            checkpoint_every=args.checkpoint_every,
        )

    logger.info(f"done: {stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
