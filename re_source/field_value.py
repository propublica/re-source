"""What a field value means, and how a chunk carries it."""

from __future__ import annotations

import json

# Some imports serialize multi-value fields as JSON arrays ('["No"]', '[null]', '[]').
# A value that unwraps to nothing / a null-like token carries no content — a chunk
# built from it is just the question label + nothing, which pollutes retrieval and
# wastes LLM validation. Genuine short answers ("No"/"Yes") are NOT blank.
_BLANK_VALUE_TOKENS = {"", "n/a", "na", "none", "null", "nil", "nan", "-", "no answer"}


def is_blank_value(value) -> bool:
    """True if a field value has no substantive content, so its chunk should be skipped.

    Returns:
        True for empty / null / 'N/A'-style tokens, after unwrapping JSON
        array/object wrappers. Genuine short answers like 'No'/'Yes' are not blank.
    """
    if value is None:
        return True
    s = str(value).strip()
    if not s:
        return True
    if s[0] in "[{":
        try:
            parsed = json.loads(s)
        except (ValueError, TypeError):
            parsed = None
        if isinstance(parsed, list):
            kept = [p for p in parsed if p is not None and str(p).strip()]
            return not kept or all(str(p).strip().lower() in _BLANK_VALUE_TOKENS for p in kept)
        if isinstance(parsed, dict):
            return not parsed
    return s.lower() in _BLANK_VALUE_TOKENS


def label_for(column_name: str) -> str:
    """The label a chunk's text leads with, for this column."""
    col = (column_name or "").strip()
    return col if col.endswith("?") else (f"{col}:" if col else "")


def with_label(column_name: str, value) -> str:
    """A chunk's text: the question, then the answer."""
    label = label_for(column_name)
    answer = str(value).strip()
    return f"{label} {answer}" if label else answer


def without_label(text: str, column_name: str) -> str:
    """The answer inside a chunk's text — `with_label` read backwards."""
    text = text or ""
    col = (column_name or "").strip()
    if not col:
        return text
    label = label_for(column_name)
    for prefix in (f"{label} ", f"{col} "):
        if prefix.strip() and text.startswith(prefix):
            return text[len(prefix):]
    return text
