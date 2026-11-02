"""Atomic chunking: split database rows into field-level chunks for precise retrieval."""

import importlib
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterator, Optional, TYPE_CHECKING

from .field_value import is_blank_value, with_label
from . import config

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .db_loader import Document


@dataclass
class AtomicChunk:
    """A single field from a database row, ready for embedding."""

    id: str  # "{parent_id}_{column_key}"
    parent_id: str  # Document ID (table + record ID)
    column_name: str  # Original column name
    column_key: str  # Normalized key (lowercase, underscores)
    text: str  # "{column_name}: {value}" for embedding
    source_file: str


@dataclass
class ParentRow:
    """The full row data for result aggregation."""

    id: str  # Document ID (table + record ID)
    source_file: str
    fields: dict[str, str]  # All original fields


def normalize_column_key(col_name: str) -> str:
    """Create a normalized key from column name for matching."""
    key = col_name.lower()
    key = re.sub(r"[^\w\s]", "", key)
    key = re.sub(r"\s+", "_", key.strip())
    return key


# What a field value means, and how a chunk carries it, live in
# re_source/field_value.py — index time and query time have to agree on both, and the
# search path was importing this whole module to reach them.


def format_chunk_text(col_name: str, value: str) -> str:
    """Combine column name and value for meaningful embedding."""
    return with_label(col_name, value)


# Chunk size is a TOKEN budget, not a character one.
#
# Both embedders read a fixed number of tokens and silently discard the rest.
# e5-large-v2 is BERT-large based, so 512 tokens is an architectural ceiling no
# config raises — it sets the size. ColBERT's window is a services.xml value
# raised to match, so neither model truncates.
#
# Measured against the real corpus before this changed: chunks were split at
# 4000 characters against a 256-token ColBERT window, leaving 24.6% of ALL text
# in no vector at all — only 2.9% of chunks were truncated, but those were the
# long narrative answers. Cost of fixing it: ~1.05x attribute memory, and with
# the JSON-blank chunks dropped the index gets smaller than it was.
MAX_CHUNK_TOKENS = 512
# Room for [CLS]/[SEP] and the "passage: " prefix e5 prepends at index time.
TOKEN_MARGIN = 12
CHUNK_OVERLAP_TOKENS = 48

# Character equivalents, used only when the tokenizer is unavailable. The corpus
# measures 3.44 chars/token; 3.0 is deliberately conservative so the fallback
# under-fills rather than truncating.
FALLBACK_CHARS_PER_TOKEN = 3.0

# Kept for callers and tests that still reason in characters.
MAX_CHUNK_LENGTH = 4000
CHUNK_OVERLAP = 200

_tokenizer_cache = None
_tokenizer_loaded = False


def get_tokenizer():
    """The tokenizer Vespa itself uses, or None to fall back to characters."""
    global _tokenizer_cache, _tokenizer_loaded
    if _tokenizer_loaded:
        return _tokenizer_cache
    _tokenizer_loaded = True
    try:
        from tokenizers import Tokenizer

        from .config import DATA_DIR

        path = DATA_DIR / "vespa_models" / "colbert" / "tokenizer.json"
        if path.exists():
            _tokenizer_cache = Tokenizer.from_file(str(path))
        else:
            logger.warning(f"{path} missing; splitting on characters instead")
    except ImportError:
        logger.warning("tokenizers not installed; splitting on characters instead")
    except Exception as e:
        logger.warning(f"could not load tokenizer ({e}); splitting on characters")
    return _tokenizer_cache


def count_tokens(text: str) -> int:
    tok = get_tokenizer()
    if tok is None:
        return int(len(text) / FALLBACK_CHARS_PER_TOKEN)
    return len(tok.encode(text, add_special_tokens=False).ids)


def split_by_tokens(text: str, max_tokens: int, overlap_tokens: int = CHUNK_OVERLAP_TOKENS) -> list[str]:
    """Split so every part fits inside `max_tokens`."""
    if max_tokens <= 0:
        return [text]
    tok = get_tokenizer()
    if tok is None:
        chars = int(max_tokens * FALLBACK_CHARS_PER_TOKEN)
        return split_long_text(text, chars, int(overlap_tokens * FALLBACK_CHARS_PER_TOKEN))

    encoding = tok.encode(text, add_special_tokens=False)
    offsets = encoding.offsets
    if len(offsets) <= max_tokens:
        return [text]

    step = max(max_tokens - overlap_tokens, 1)
    parts = []
    start_tok = 0
    while start_tok < len(offsets):
        end_tok = min(start_tok + max_tokens, len(offsets))
        start_char = offsets[start_tok][0]
        end_char = offsets[end_tok - 1][1] if end_tok else len(text)
        part = text[start_char:end_char].strip()
        if part:
            parts.append(part)
        if end_tok >= len(offsets):
            break
        start_tok += step
    return parts or [text]


def split_long_text(text: str, max_length: int = MAX_CHUNK_LENGTH, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split long text into overlapping chunks.

    Args:
        text: The text to split.
        max_length: Maximum characters per chunk.
        overlap: Characters to overlap between chunks.

    Returns:
        List of text chunks (single item if text is short enough).
    """
    if len(text) <= max_length:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_length, len(text))
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap

    return chunks


# How a column is chunked. An optional reviewed `field_inventory` table keeps
# classification in one place, with a human override stored alongside any
# suggestion.
#
#   answer_only    free text — the answer stands alone, so the question is dead
#                  weight: it burns tokens inside a fixed window, and being
#                  identical across every answer in the column it makes any query
#                  echoing the question match all of them
#   with_question  closed-ended or unclear — "No" alone is meaningless
#
# The question is never lost: column_name is a separate attribute on the chunk,
# already requested in _CHUNK_FIELDS and already what result cards render.
_ANSWER_ONLY_CLASSES = {"content_free_text"}

_field_classes_cache: Optional[dict[tuple[str, str], str]] = None
_reviewed_field_classes_cache: Optional[set[tuple[str, str]]] = None


def load_field_classes() -> dict[tuple[str, str], str]:
    """(source_file, column_name) -> classification, from the field_inventory table.

    Returns:
        (source_file, column_name) -> classification, preferring the reviewed
        `classification` over `suggested_class`. {} when the table doesn't exist.
    """
    global _field_classes_cache, _reviewed_field_classes_cache
    if _field_classes_cache is not None:
        return _field_classes_cache

    import duckdb

    from .config import DUCKDB_PATH

    classes: dict[tuple[str, str], str] = {}
    reviewed_classes: set[tuple[str, str]] = set()
    if not DUCKDB_PATH.exists():
        _field_classes_cache = classes
        _reviewed_field_classes_cache = reviewed_classes
        return classes
    try:
        conn = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    except Exception as e:
        logger.warning(f"field_inventory unavailable ({e}); using unclassified defaults")
        _field_classes_cache = classes
        _reviewed_field_classes_cache = reviewed_classes
        return classes
    try:
        exists = conn.execute(
            "SELECT count(*) FROM information_schema.tables "
            "WHERE table_name = 'field_inventory'"
        ).fetchone()[0]
        if exists:
            for source_file, column, reviewed, suggested in conn.execute(
                "SELECT source_file, column_name, classification, suggested_class "
                "FROM field_inventory"
            ).fetchall():
                cls = reviewed or suggested
                if not cls:
                    continue
                key = (source_file, column)
                classes[key] = cls
                if reviewed is not None:
                    reviewed_classes.add(key)
    except Exception as e:
        logger.warning(f"could not read field_inventory ({e}); using unclassified defaults")
    finally:
        conn.close()

    if classes:
        logger.info(f"field_inventory: {len(classes)} classified columns")
    _field_classes_cache = classes
    _reviewed_field_classes_cache = reviewed_classes
    return classes


def reviewed_field_classes() -> set[tuple[str, str]]:
    """Field classifications explicitly reviewed by a person, not suggestions."""
    if _field_classes_cache is None:
        load_field_classes()
    return _reviewed_field_classes_cache or set()


def _public_chunk_mode(source_file: str, col_name: str) -> str:
    """Public default: retain every user field, shortening known prose fields."""
    cls = load_field_classes().get((source_file, col_name))
    if cls in _ANSWER_ONLY_CLASSES:
        return "answer_only"
    return "with_question"


@lru_cache(maxsize=8)
def _load_chunk_policy(module_name: str):
    module = importlib.import_module(module_name)
    resolver = getattr(module, "chunk_mode", None)
    if not callable(resolver):
        raise ValueError(
            f"CHUNK_POLICY_MODULE {module_name!r} must define chunk_mode(source_file, column_name)"
        )
    return resolver


def chunk_mode(source_file: str, col_name: str) -> str:
    """How to embed one user field, optionally delegated to an operator policy.

    Public Re-Source has no field-exclusion policy: every non-internal field is
    chunked. Deployments that need one provide a module through
    ``CHUNK_POLICY_MODULE``. Its ``chunk_mode`` function returns ``skip``,
    ``answer_only``, or ``with_question``.
    """
    module_name = config.CHUNK_POLICY_MODULE.strip()
    resolver = _load_chunk_policy(module_name) if module_name else _public_chunk_mode
    mode = resolver(source_file, col_name)
    if mode not in {"skip", "answer_only", "with_question"}:
        raise ValueError(
            f"Chunk policy returned invalid mode {mode!r} for {source_file}.{col_name}"
        )
    return mode


# Optional reviewed yes/no clauses supplied by corpus-maintenance tooling.
#
# "Are you a teacher? Yes" and "...? No" differ by one token in six, so they
# embed almost on top of each other — dense retrieval cannot separate a claim
# from its denial. The fix is entirely in the text: each answer becomes a clause
# stating its own positive fact ("went without a school lunch", never "did not
# receive a school lunch"). Retrieval then needs no filter and no special case,
# because the embedding never has to represent "not".
_YES = {"yes", "y", "true", "checked", "1"}
_NO = {"no", "n", "false", "unchecked", "0"}

_templates_cache: Optional[dict[tuple[str, str], dict]] = None


def load_question_templates() -> dict[tuple[str, str], dict]:
    """(source_file, column_name) -> {yes, no, yes_skip, no_skip}, REVIEWED rows only."""
    global _templates_cache
    if _templates_cache is not None:
        return _templates_cache

    import duckdb

    from .config import DUCKDB_PATH

    out: dict[tuple[str, str], dict] = {}
    if not DUCKDB_PATH.exists():
        _templates_cache = out
        return out
    try:
        conn = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    except Exception as e:
        logger.warning(f"question_templates unavailable ({e})")
        _templates_cache = out
        return out
    try:
        exists = conn.execute(
            "SELECT count(*) FROM information_schema.tables "
            "WHERE table_name = 'question_templates'"
        ).fetchone()[0]
        if exists:
            for sf, col, yes, no, yskip, nskip in conn.execute(
                "SELECT source_file, column_name, yes_clause, no_clause, "
                "yes_skip, no_skip FROM question_templates WHERE reviewed"
            ).fetchall():
                out[(sf, col)] = {"yes": yes or "", "no": no or "",
                                  "yes_skip": bool(yskip), "no_skip": bool(nskip)}
    except Exception as e:
        logger.warning(f"could not read question_templates ({e})")
    finally:
        conn.close()

    if out:
        logger.info(f"question_templates: {len(out)} reviewed questions")
    _templates_cache = out
    return out


def templated_text(source_file: str, col_name: str, value) -> Optional[str]:
    """The reviewed clause for this answer, "" to skip the chunk entirely, or
    None when no reviewed template applies and normal chunking should run."""
    tpl = load_question_templates().get((source_file, col_name))
    if tpl is None:
        return None
    answer = str(value).strip().strip('[]"\'').lower()
    if answer in _YES:
        return "" if tpl["yes_skip"] else (tpl["yes"] or None)
    if answer in _NO:
        return "" if tpl["no_skip"] else (tpl["no"] or None)
    # A value outside the yes/no vocabulary means the column is not what the
    # template assumed; fall through rather than mislabel it.
    return None


def chunk_document(doc: "Document") -> tuple[list[AtomicChunk], ParentRow]:
    """Split a Document (from db_loader) into atomic chunks.

    Args:
        doc: Document object from db_loader

    Returns:
        Tuple of (list of AtomicChunks, ParentRow)
    """
    # Use document ID as parent_id (already includes table + record ID)
    parent_id = doc.id
    source = doc.source_file

    chunks = []

    for col_name, value in doc.fields.items():
        # Skip internal/metadata fields
        if col_name.startswith("_"):
            continue

        # Indexed at all, and does the answer need its question? See chunk_mode.
        mode = chunk_mode(source, col_name)
        if mode == "skip":
            continue

        # Skip values with no substantive content — empty, or JSON-wrapped
        # null/N-A/empty like '[null]' / '["N/A"]' / '[]'. A chunk from one of
        # these is just the question label with no answer (retrieval noise).
        if is_blank_value(value):
            continue

        col_key = normalize_column_key(col_name)

        # A reviewed yes/no clause replaces the text entirely — "" means the
        # reviewer marked this answer as carrying nothing worth indexing.
        clause = templated_text(source, col_name, value)
        if clause == "":
            continue
        raw = clause if clause is not None else str(value).strip()

        # Split the answer to fit the embedding window.
        budget = MAX_CHUNK_TOKENS - TOKEN_MARGIN
        text_parts = split_by_tokens(raw, budget)

        # The question rides along only on an answer that fits in one chunk. An
        # answer long enough to split is long enough to stand on its own, so its
        # parts are chunked bare — repeating the question across them would
        # spend the window on text nobody wrote. (A with_question column that
        # splits is also a hint the column is misclassified.)
        if clause is None and mode == "with_question" and len(text_parts) == 1:
            with_q = format_chunk_text(col_name, text_parts[0])
            # Only if it still fits; a verbose question plus a middling answer
            # can overflow, and then the answer stands alone too.
            if count_tokens(with_q) <= budget:
                text_parts = [with_q]

        for part_idx, part_text in enumerate(text_parts):
            # Add part index to ID if split
            if len(text_parts) > 1:
                chunk_id = f"{parent_id}_{col_key}_p{part_idx}"
            else:
                chunk_id = f"{parent_id}_{col_key}"

            chunk = AtomicChunk(
                id=chunk_id,
                parent_id=parent_id,
                column_name=col_name,
                column_key=col_key,
                text=part_text,
                source_file=source,
            )
            chunks.append(chunk)

    parent = ParentRow(
        id=parent_id,
        source_file=source,
        fields={k: str(v) for k, v in doc.fields.items() if v},
    )

    return chunks, parent


def chunk_documents(docs: list["Document"]) -> Iterator[tuple[list[AtomicChunk], ParentRow]]:
    """Chunk all documents from db_loader.

    Args:
        docs: List of Document objects from db_loader

    Yields:
        Tuple of (list of AtomicChunks, ParentRow) for each document
    """
    total_chunks = 0

    for doc in docs:
        chunks, parent = chunk_document(doc)

        # Only yield if we got at least one chunk
        if chunks:
            total_chunks += len(chunks)
            yield chunks, parent

    logger.info(f"Total: {len(docs)} documents -> {total_chunks} chunks")
