"""Shared XLSX export with summary tab and name/email detection."""

import importlib
import json
import re
from functools import lru_cache
from io import BytesIO
import pandas as pd

from . import config


# ---------------------------------------------------------------------------
# Name / email column detection
# ---------------------------------------------------------------------------
# Priority-ordered patterns for person name columns (highest priority first).
# These match the respondent's name, NOT facility/school/company names.
_NAME_PATTERNS = [
    re.compile(r"^name$", re.I),
    re.compile(r"^your name", re.I),
    re.compile(r"^respondent name", re.I),
    re.compile(r"^source name$", re.I),
    re.compile(r"^tipster name$", re.I),
    re.compile(r"^what is your name", re.I),
    re.compile(r"^name \(from", re.I),
    re.compile(r"^name(?: \d+)? \(from", re.I),
    re.compile(r"^source name", re.I),
    re.compile(r"name \(from link", re.I),
    re.compile(r"^name of person", re.I),
]

_EMAIL_PATTERNS = [
    re.compile(r"^email$", re.I),
    re.compile(r"^e-?mail$", re.I),
    re.compile(r"^your .*email", re.I),
    re.compile(r"^respondent email", re.I),
    re.compile(r"^what is your email", re.I),
    re.compile(r"^email \(from", re.I),
    re.compile(r"^email(?: \d+)? \(from", re.I),
    re.compile(r"email \(from link", re.I),
]


def _norm_key(name: str) -> str:
    """Column name reduced to a comparable key: lowercased, '_'/'-' read as spaces, runs of
    whitespace collapsed.
    """
    return re.sub(r"\s+", " ", re.sub(r"[_-]+", " ", name).strip().lower())


def _find_column(columns: list[str], patterns: list[re.Pattern]) -> str | None:
    """Return the first column matching any pattern, in priority order."""
    for pattern in patterns:
        for col in columns:
            if pattern.search(col):
                return col
    return None


# ---------------------------------------------------------------------------
# Vespa search export — enriched with record URL, dataset name, chunk text
# ---------------------------------------------------------------------------

_EXCEL_FORBIDDEN = re.compile(r"[\[\]:*?/\\]")


def _safe_sheet_name(name: str, used: set[str]) -> str:
    """Excel-legal, <=31 char, unique sheet name (strips []:*?/\\, de-dupes)."""
    clean = _EXCEL_FORBIDDEN.sub("_", name).strip() or "Sheet"
    clean = clean[:31]
    if clean not in used:
        used.add(clean)
        return clean
    # De-dupe with a numeric suffix, keeping within 31 chars.
    for i in range(2, 1000):
        suffix = f"_{i}"
        candidate = clean[: 31 - len(suffix)] + suffix
        if candidate not in used:
            used.add(candidate)
            return candidate
    used.add(clean)
    return clean


def _clean_value(val) -> str:
    """Unwrap list/JSON-array values into a plain display string."""
    if val is None:
        return ""
    s = str(val).strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return ", ".join(str(x) for x in parsed if x is not None)
        except (json.JSONDecodeError, ValueError):
            pass
    return s


# openpyxl refuses to write these outright (IllegalCharacterError): the C0
# control characters, minus the three xlsx allows (tab, newline, carriage
# return). They arrive in tip text from pasted documents and PDFs, and a single
# one anywhere in the corpus used to abort the entire export.
_ILLEGAL_XLSX_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _strip_illegal_chars(val):
    """Drop characters xlsx cannot hold.

    Returns:
        The cleaned string. Non-strings pass through untouched.
    """
    return _ILLEGAL_XLSX_CHARS.sub("", val) if isinstance(val, str) else val


def _sanitized(df: pd.DataFrame) -> pd.DataFrame:
    """Copy of `df` with every string cell made writable."""
    return df.map(_strip_illegal_chars).rename(columns=_strip_illegal_chars)


def _neutralize_formulas(worksheet) -> int:
    """Force every formula-typed cell in `worksheet` back to a string."""
    fixed = 0
    for row in worksheet.iter_rows():
        for cell in row:
            if cell.data_type == "f":
                cell.data_type = "s"
                fixed += 1
    return fixed


def _matching_chunks_text(payload: dict) -> str:
    """Full matched-chunk text: one 'column_name: text' per line."""
    chunks = (payload or {}).get("matching_chunks", [])
    return "\n".join(
        f"{c.get('column_name', '')}: {c.get('text', '')}" for c in chunks
    )


def _cited_evidence_text(payload: dict) -> str:
    """The field/quote pairs a validator cited: one 'field: quote' per line."""
    evidence = (payload or {}).get("evidence") or []
    return "\n".join(f"{e.get('field', '')}: {e.get('quote', '')}" for e in evidence)


@lru_cache(maxsize=8)
def _load_summary_columns(module_name: str):
    module = importlib.import_module(module_name)
    resolver = getattr(module, "summary_columns", None)
    if not callable(resolver):
        raise ValueError(
            f"EXPORT_POLICY_MODULE {module_name!r} must define summary_columns()"
        )
    return resolver


def summary_columns() -> list[tuple[str, str | tuple[str, ...]]]:
    """(header, field or fields) pairs added to the Summary tab, from ``EXPORT_POLICY_MODULE``."""
    # Public Re-Source adds none; a deployment names its own corpus's columns.
    module_name = config.EXPORT_POLICY_MODULE.strip()
    return list(_load_summary_columns(module_name)()) if module_name else []


def _summary_value(by_key: dict, fields: str | tuple[str, ...]) -> str:
    """One policy column's cell: the distinct non-empty values of its fields, joined."""
    values = []
    for field in (fields,) if isinstance(fields, str) else fields:
        value = _clean_value(by_key.get(_norm_key(field)))
        if value and value not in values:
            values.append(value)
    return "; ".join(values)


def _build_vespa_summary(results: list) -> pd.DataFrame:
    """Summary rows for the Vespa export: one row per result, enriched with dataset
    name, record id, record URL, full chunk text, and the LLM's reason."""
    from .datasets import dataset_label, record_url

    # Detect name/email columns per source (same approach as the semantic export).
    by_source: dict[str, set] = {}
    for r in results:
        by_source.setdefault(r.source_file, set()).update((r.fields or {}).keys())
    source_name_col = {s: _find_column(sorted(cols), _NAME_PATTERNS) for s, cols in by_source.items()}
    source_email_col = {s: _find_column(sorted(cols), _EMAIL_PATTERNS) for s, cols in by_source.items()}

    extra_columns = summary_columns()
    rows = []
    for i, r in enumerate(results, 1):
        fields = r.fields or {}
        payload = r.payload or {}
        by_key = {_norm_key(k): v for k, v in fields.items()}
        record_id = fields.get("_record_id", "")
        name_col = source_name_col.get(r.source_file)
        email_col = source_email_col.get(r.source_file)
        rows.append(
            {
                "ID": i,
                "Dataset": dataset_label(r.source_file),
                "Dataset ID": r.source_file,
                "Record ID": record_id,
                "Record URL": record_url(r.source_file, record_id) or "",
                "Name": _clean_value(fields.get(name_col)) if name_col else "",
                "Email": _clean_value(fields.get(email_col)) if email_col else "",
                "Submitted Date": fields.get("_created_time", ""),
                # Matched however the dataset spells the field (case, _ or -).
                **{header: _summary_value(by_key, fields)
                   for header, fields in extra_columns},
                # Only a batch run over several queries fills `queries` (see
                # scripts/author_batch_export.py); a single-query export leaves
                # the column blank rather than repeating the query on every row.
                "Matched Queries": "; ".join(payload.get("queries", [])),
                "Matched Fields": ", ".join(
                    c["column_name"] for c in payload.get("matching_chunks", [])
                ),
                "Matching Chunks": _matching_chunks_text(payload),
                "Cited Evidence": _cited_evidence_text(payload),
                "AI Reason": payload.get("llm_reason", ""),
            }
        )
    return pd.DataFrame(rows)


def export_vespa_xlsx(results: list) -> BytesIO:
    """Export Vespa search results to XLSX: a 'Summary' tab with everything
    (dataset name, record URL, LLM reasoning, full chunk text) plus one sheet per
    dataset holding the full rows, each prefixed with its record URL."""
    from .datasets import dataset_label, record_url

    # Group full rows by dataset, carrying a record URL column.
    by_dataset: dict[str, list[dict]] = {}
    for r in results:
        fields = r.fields or {}
        record_id = fields.get("_record_id", "")
        row = {
            "Record URL": record_url(r.source_file, record_id) or "",
            **{k: _clean_value(v) for k, v in fields.items()},
        }
        by_dataset.setdefault(r.source_file, []).append(row)

    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        _sanitized(_build_vespa_summary(results)).to_excel(
            writer, sheet_name="Summary", index=False
        )

        used_names = {"Summary"}
        for dataset_id, field_rows in by_dataset.items():
            sheet = _safe_sheet_name(dataset_label(dataset_id), used_names)
            _sanitized(pd.DataFrame(field_rows)).to_excel(
                writer, sheet_name=sheet, index=False
            )

        # Neutralize formula injection from tip text, after every sheet is
        # written but while the workbook is still open.
        for worksheet in writer.sheets.values():
            _neutralize_formulas(worksheet)

    output.seek(0)
    return output


def _query_tab_row(result) -> dict:
    """One record's row in a per-query tab: link, that query's verdict, then its own fields."""
    from .datasets import record_url

    fields = result.fields or {}
    payload = result.payload or {}
    row = {
        "Record URL": record_url(result.source_file, fields.get("_record_id", "")) or "",
        "Why Matched": payload.get("llm_reason", ""),
        "Cited Evidence": _cited_evidence_text(payload),
    }
    for key, value in fields.items():
        row.setdefault(key, _clean_value(value))
    return row


def export_by_query_xlsx(merged: list, per_query: list[tuple[str, list]]) -> BytesIO:
    """Export a batch run: a 'Summary' tab of every record, then one tab per query
    holding one table per dataset (datasets have different columns)."""
    from openpyxl.styles import Font

    from .datasets import dataset_label

    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        _sanitized(_build_vespa_summary(merged)).to_excel(
            writer, sheet_name="Summary", index=False
        )

        used_names = {"Summary"}
        for label, results in per_query:
            if not results:
                continue
            sheet = _safe_sheet_name(label, used_names)
            by_dataset: dict[str, list] = {}
            for r in sorted(results, key=lambda r: r.score, reverse=True):
                by_dataset.setdefault(r.source_file, []).append(r)

            start = 0
            for dataset_id, rows in by_dataset.items():
                noun = "match" if len(rows) == 1 else "matches"
                title = f"{dataset_label(dataset_id)} ({len(rows)} {noun})"
                _sanitized(pd.DataFrame([[title]])).to_excel(
                    writer, sheet_name=sheet, startrow=start, index=False, header=False
                )
                writer.sheets[sheet].cell(row=start + 1, column=1).font = Font(bold=True)
                _sanitized(pd.DataFrame([_query_tab_row(r) for r in rows])).to_excel(
                    writer, sheet_name=sheet, startrow=start + 1, index=False
                )
                # Title, header, rows, then two blank rows before the next table.
                start += len(rows) + 4

        for worksheet in writer.sheets.values():
            _neutralize_formulas(worksheet)

    output.seek(0)
    return output
