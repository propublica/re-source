"""What one result shows, worked out once for both views."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from .datasets import dataset_label, record_url
from .export import _clean_value
from .field_value import without_label
from .export_selection import identity


@dataclass(frozen=True)
class Match:
    """One field a result matched in: the question, and the answer to it."""

    question: str
    answer: str


@dataclass(frozen=True)
class ResultCard:
    """Everything either view needs about one result."""

    index: int
    ident: str
    dataset: str
    submitted: date | None
    url: str | None
    matches: tuple[Match, ...]
    fallback: str
    matched_fields: str
    highlights: tuple[str, ...]
    why: str
    evidence_label: str
    included: bool
    fields: dict


# What a "why this matched" box says when the strategy didn't produce one — BM25
# does no LLM pass at all.
NOTHING_TO_SAY = "—"


def dom_ident(result, index: int) -> str:
    """A DOM-safe identity for this result's keyed containers."""
    ident = identity(result)
    if ident is None:
        return f"unidentified-{index}"
    return re.sub(r"\W+", "-", ident).strip("-")


def chunk_answer(text: str, column_name: str) -> str:
    """The answer inside a chunk's text, ready to show."""
    return _clean_value(without_label(text, column_name))


def submitted_date(result) -> date | None:
    """When the tip was filed, from the row's `_created_time`."""
    raw = (getattr(result, "fields", None) or {}).get("_created_time") or ""
    if not raw:
        return None
    try:
        from datetime import datetime

        return datetime.fromisoformat(raw.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def card_for(index: int, result, selection) -> ResultCard:
    """Everything either view needs to draw `result`, derived once."""
    payload = result.payload or {}
    evidence = payload.get("evidence")
    if evidence is not None:
        chunks = [{"column_name": e.get("field"), "text": e.get("quote")} for e in evidence]
    else:
        chunks = payload.get("matching_chunks") or []

    matches = tuple(
        Match(
            question=c.get("column_name") or "",
            answer=chunk_answer(c.get("text") or "", c.get("column_name") or ""),
        )
        for c in chunks
    )
    named = [c.get("column_name") or "" for c in chunks[:5]]
    fields = result.fields or {}

    return ResultCard(
        index=index,
        ident=dom_ident(result, index),
        dataset=dataset_label(result.source_file),
        submitted=submitted_date(result),
        url=record_url(result.source_file, fields.get("_record_id")),
        matches=matches,
        fallback=_clean_value(result.text or ""),
        matched_fields=", ".join(n for n in named if n) or NOTHING_TO_SAY,
        highlights=tuple(payload.get("highlights") or ()),
        why=payload.get("llm_reason") or NOTHING_TO_SAY,
        # The Author Profile strategy shows the quotes the LLM cited (`evidence`)
        # rather than search matches. Saying which it is rather than making each
        # view sniff a key for None.
        evidence_label=("Matching evidence" if payload.get("evidence") is not None
                        else "Top matching chunks"),
        included=selection.includes(result),
        fields=fields,
    )
