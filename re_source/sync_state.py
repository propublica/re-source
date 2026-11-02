"""Track indexed rows for incremental embedding."""

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .config import VESPA_SYNC_STATE_PATH

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .db_loader import Document


@dataclass
class SyncState:
    """Tracks which documents have been indexed and their content hashes."""

    # Maps doc_id -> content_hash
    indexed: dict[str, str] = field(default_factory=dict)

    # Metadata
    last_sync_timestamp: str = ""
    total_indexed_count: int = 0


def load_sync_state(path: Path | None = None) -> SyncState:
    """Load sync state from disk.

    Args:
        path: Optional override of the state file path. Defaults to
            VESPA_SYNC_STATE_PATH. Callers (embed_vespa, enrich) pass an
            explicit path for their own pipeline's state.

    Returns:
        SyncState object (empty if file doesn't exist)
    """
    p = path or VESPA_SYNC_STATE_PATH
    if not p.exists():
        return SyncState()

    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not load sync state at {p}") from exc

    if not isinstance(data, dict) or not isinstance(data.get("indexed", {}), dict):
        raise RuntimeError(f"Invalid sync state at {p}")
    return SyncState(
        indexed=data.get("indexed", {}),
        last_sync_timestamp=data.get("last_sync_timestamp", ""),
        total_indexed_count=data.get("total_indexed_count", 0),
    )


def save_sync_state(state: SyncState, path: Path | None = None) -> None:
    """Save sync state to disk.

    Args:
        state: The SyncState to save
        path: Optional override of the state file path (see load_sync_state).
    """
    p = path or VESPA_SYNC_STATE_PATH
    p.parent.mkdir(parents=True, exist_ok=True)

    from datetime import datetime

    state.last_sync_timestamp = datetime.now().isoformat()
    state.total_indexed_count = len(state.indexed)

    data = {
        "indexed": state.indexed,
        "last_sync_timestamp": state.last_sync_timestamp,
        "total_indexed_count": state.total_indexed_count,
    }

    fd, temporary = tempfile.mkstemp(prefix=f".{p.name}.", suffix=".tmp", dir=p.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, p)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def filter_to_changed(docs: list["Document"], state: SyncState) -> list["Document"]:
    """Return only documents that are new or changed.

    Args:
        docs: All documents from the database
        state: Current sync state with indexed hashes

    Returns:
        List of documents that need to be indexed (new or changed)
    """
    changed = []
    for doc in docs:
        existing_hash = state.indexed.get(doc.id)
        if existing_hash != doc.content_hash:
            changed.append(doc)
    return changed


def get_deleted_doc_ids(docs: list["Document"], state: SyncState) -> set[str]:
    """Find documents that were indexed but no longer exist.

    Args:
        docs: All current documents from the database
        state: Current sync state

    Returns:
        Set of doc_ids that should be removed from the index
    """
    current_ids = {doc.id for doc in docs}
    indexed_ids = set(state.indexed.keys())
    return indexed_ids - current_ids


def update_state_for_docs(state: SyncState, docs: list["Document"]) -> None:
    """Update sync state with newly indexed documents.

    Args:
        state: SyncState to update
        docs: Documents that were indexed
    """
    for doc in docs:
        state.indexed[doc.id] = doc.content_hash


def remove_from_state(state: SyncState, doc_ids: set[str]) -> None:
    """Remove deleted documents from sync state.

    Args:
        state: SyncState to update
        doc_ids: Document IDs to remove
    """
    for doc_id in doc_ids:
        state.indexed.pop(doc_id, None)
