"""Streamlit render for the admin data-upload page."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st

from re_source.access import is_admin
from re_source.db_loader import (
    all_tables,
    column_names,
    content_hashes,
    count_rows,
    list_tables,
    record_ids,
)
from re_source.importer import import_dataframe, read_file
from re_source.search_ui import render_header

logger = logging.getLogger(__name__)

# Session-state keys, namespaced so they can't collide with the search pages'.
_PARSED = "admin::parsed"
_IMPORTED = "admin::imported"
_DELTA = "admin::delta"

_PREVIEW_ROWS = 8


# The rules that judge an upload, and the plan that sequences them, live in
# re_source/import_plan.py. Only the names this page actually draws with are
# imported — the rest used to be re-exported so tests could reach them here,
# which gave "what an upload may do" two front doors.
from re_source.import_plan import (  # noqa: E402
    _friendly_db_error,
    id_column_candidates,
    suggest_table_name,
    uploadable_tables,
)
from re_source.import_plan import plan as build_import_plan  # noqa: E402


class _DuckDBView:
    """The database reads a plan needs, over this project's DuckDB."""

    def list_tables(self):
        return list_tables()

    def all_tables(self):
        return all_tables()

    def record_ids(self, table):
        return record_ids(table)

    def count_rows(self, table):
        return count_rows(table)

    def content_hashes(self, table):
        return content_hashes(table)

    def column_names(self, table):
        return column_names(table)


def _forget_plan() -> None:
    """Drop the memoised plan."""
    st.session_state.pop(_DELTA, None)


def _cached_plan(df, id_column: str, table: str, file_key):
    """The import plan, memoised against what it was computed for."""
    cache_key = (table, id_column, file_key)
    cached = st.session_state.get(_DELTA)
    if cached and cached["key"] == cache_key:
        return cached["value"]

    value = build_import_plan(df, id_column=id_column, table=table, db=_DuckDBView())
    st.session_state[_DELTA] = {"key": cache_key, "value": value}
    return value


# ----- Streamlit page -------------------------------------------------------

def _inject_css() -> None:
    st.markdown(
        """
        <style>
        /* Roomy stacked cards, matching the search pages' keyed-container
           convention rather than a dense table. */
        [class*="st-key-admin-card"] {
            padding: 0.25rem 0.25rem 0.5rem 0.25rem;
        }
        [class*="st-key-admin-card"] hr {
            margin: 0.5rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _sheet_names(path: Path) -> list[str]:
    """Sheet names in an XLSX, or [] for a CSV / unreadable workbook."""
    if path.suffix.lower() not in (".xlsx", ".xls"):
        return []
    try:
        with pd.ExcelFile(path) as book:
            return list(book.sheet_names)
    except Exception as exc:
        logger.warning(f"could not read sheet names from {path}: {exc}")
        return []


def _parse_upload(uploaded, sheet: str | None) -> tuple[pd.DataFrame, list[str]]:
    """Write the uploaded bytes to a temp file and parse it with the same reader the CLI
    uses.

    The temp file holds tip data, so it exists only for the duration of the
    parse. The path is captured before anything is written to it: binding it
    after the write left it unbound on failure, and the file on disk forever.

    Returns:
        (frame, sheet_names).
    """
    suffix = Path(uploaded.name).suffix or ".csv"
    handle = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp = Path(handle.name)
    try:
        with handle:
            handle.write(uploaded.getbuffer())
        sheets = _sheet_names(tmp)
        df = read_file(tmp, sheet)
        return df, sheets
    finally:
        tmp.unlink(missing_ok=True)


def _render_existing_tables() -> None:
    """What's already loaded, so an uploader can see whether they're adding a
    table or replacing one."""
    tables = uploadable_tables(list_tables())
    if not tables:
        st.info("No uploaded files yet.")
        return

    with st.container(border=True, key="admin-card-tables"):
        st.markdown("**Uploaded files**")
        for name in tables:
            left, right = st.columns([3, 1], vertical_alignment="center")
            left.markdown(f"`{name}`")
            try:
                right.caption(f"{count_rows(name):,} rows")
            except duckdb.Error:
                right.caption("unreadable")


def _render_feed_step(table: str) -> None:
    """Step 2: push the imported table into Vespa."""
    with st.container(border=True, key="admin-card-feed"):
        st.markdown("**Step 2 — index for search**")
        st.caption(
            f"`{table}` is in the database but not yet searchable. Indexing "
            "computes embeddings inside Vespa and can take several minutes for a "
            "large table — leave this tab open."
        )
        if not st.button("Index to Vespa", type="primary", key="admin-feed-btn"):
            return

        # Imported lazily: this pulls the chunker and tokenizers, which the
        # search pages don't need, so it stays off the app's import path.
        from re_source.embed_vespa import feed_source_file
        from re_source.vespa_client import get_vespa_client

        with st.status(f"Indexing `{table}`…", expanded=True) as status:
            try:
                stats = feed_source_file(
                    table, get_vespa_client(), record_state=True
                )
            except Exception as exc:
                logger.exception("feed failed")
                status.update(label="Indexing failed", state="error")
                st.error(f"Indexing failed: {exc}")
                return

            failed = stats["parent_failures"] or stats["chunk_failures"]
            if failed:
                status.update(label="Indexing incomplete", state="error")
            else:
                status.update(label=f"Indexed `{table}`", state="complete")

        if failed:
            st.warning(
                f"{stats['parent_failures']} record and {stats['chunk_failures']} "
                "chunk feeds failed, so this table was not recorded as indexed. "
                "Click Index again to retry."
            )
            return
        st.success(
            f"Fed {stats['parents']:,} records and {stats['chunks']:,} chunks "
            f"in {stats['elapsed_seconds']}s."
        )
        st.caption("Switch to Card view and search for something in this file.")


def _run_author_enrichment(table: str) -> tuple[dict, dict]:
    """Compute and feed author profiles for one uploaded table."""
    # Imported lazily because the enrichment path pulls in the configured LLM
    # provider and Vespa client. Neither belongs on the normal upload import
    # path until the user explicitly starts this step.
    from re_source.db_loader import get_all_documents
    from re_source.enrich import enrich_rows, feed_enrichment
    from re_source.vespa_client import get_vespa_client

    docs = [doc for doc in get_all_documents() if doc.source_file == table]
    compute_stats = enrich_rows(docs)
    feed_stats = feed_enrichment(get_vespa_client(), docs)
    return compute_stats, feed_stats


def _render_enrichment_step(table: str) -> None:
    """Step 3: enrich this table for the Author Profile strategy."""
    with st.container(border=True, key="admin-card-enrichment"):
        st.markdown("**Step 3 — enrich authors for author search**")
        st.caption(
            f"Use the configured LLM to describe what each author of `{table}` "
            "reveals about themselves, then index those profiles in Vespa. "
            "Only new or changed records are processed."
        )
        if not st.button(
            "Run author enrichment", type="primary", key="admin-enrichment-btn"
        ):
            return

        with st.spinner(f"Enriching authors in `{table}`…"):
            try:
                compute, feed = _run_author_enrichment(table)
            except Exception as exc:
                logger.exception("author enrichment failed")
                st.error(f"Author enrichment failed: {exc}")
                return

            failures = compute["failed"]

        st.success(
            f"Processed {compute['scanned']:,} records: "
            f"{compute['enriched']:,} new or changed, "
            f"{compute['skipped_unchanged']:,} unchanged, and "
            f"{compute['nonempty_profiles']:,} non-empty profiles. "
            f"Fed {feed['fed_chunks']:,} author-profile chunks to Vespa."
        )
        if failures:
            st.warning(
                f"{failures:,} LLM calls failed. Run author enrichment again "
                "to retry them."
            )


def render_admin_upload() -> None:
    """Page: upload a CSV/XLSX, import it, index it, then enrich authors."""
    # Gating lives in this helper (it calls st.stop()); a page that skips it is
    # completely ungated. It also draws the menu — the only way off this page
    # now that the sidebar navigation is hidden.
    render_header("Upload data")
    if not is_admin(st.user):
        st.error("You do not have permission to upload or index data.")
        st.stop()
    _inject_css()

    st.caption(
        "Load a form export into the search index. Three steps: import the file "
        "into the database, index it for search, then enrich its authors for "
        "author search."
    )

    uploaded = st.file_uploader(
        "CSV or Excel file", type=["csv", "xlsx", "xls"], key="admin-upload"
    )

    if uploaded is None:
        st.session_state.pop(_PARSED, None)
        _render_existing_tables()
        # A previous import in this session may still be waiting to be indexed.
        if st.session_state.get(_IMPORTED):
            table = st.session_state[_IMPORTED]["table"]
            _render_feed_step(table)
            _render_enrichment_step(table)
        return

    _render_import_step(uploaded)

    if st.session_state.get(_IMPORTED):
        table = st.session_state[_IMPORTED]["table"]
        _render_feed_step(table)
        _render_enrichment_step(table)

    _render_existing_tables()


def _render_import_step(uploaded) -> None:
    """Step 1: parse, preview, validate, write to DuckDB."""
    file_key = (uploaded.name, uploaded.size)
    cached = st.session_state.get(_PARSED) or {}
    same_file = cached.get("key") == file_key
    # `df: None` is how the sheet picker below asks for a re-parse.
    sheet = cached.get("sheet") if same_file else None

    try:
        if same_file and cached.get("df") is not None:
            df, sheets = cached["df"], cached["sheets"]
        else:
            df, sheets = _parse_upload(uploaded, sheet)
    except Exception as exc:
        logger.exception("could not parse upload")
        st.error(f"Could not read that file: {exc}")
        return

    st.session_state[_PARSED] = {
        "key": file_key, "sheet": sheet, "df": df, "sheets": sheets
    }

    with st.container(border=True, key="admin-card-import"):
        st.markdown("**Step 1 — import into the database**")

        if len(sheets) > 1:
            picked = st.selectbox("Sheet", sheets, key="admin-sheet")
            if picked != (sheet or sheets[0]):
                # Re-parse the chosen sheet on the next run.
                st.session_state[_PARSED] = {
                    "key": file_key, "sheet": picked, "df": None, "sheets": sheets
                }
                st.rerun()

        st.caption(f"{len(df):,} rows × {len(df.columns):,} columns")
        st.dataframe(df.head(_PREVIEW_ROWS), use_container_width=True)

        if df.empty or not len(df.columns):
            st.error("That file has no rows to import.")
            return

        id_column = st.selectbox(
            "ID column",
            id_column_candidates(list(df.columns)),
            help="The column holding a stable, unique ID per row. Becomes "
                 "`_record_id` in the database.",
            key="admin-id-column",
        )
        table = suggest_table_name(uploaded.name)

        plan = _cached_plan(df, id_column, table, (file_key, sheet))

        for level, message in plan.problems:
            (st.error if level == "error" else st.warning)(message)
        if plan.blocked:
            return

        st.divider()

        confirmed = True
        if plan.needs_confirmation:
            st.info(plan.notice)
            st.caption(plan.change_summary)
            if plan.column_note:
                st.caption(plan.column_note)
            confirmed = st.checkbox(plan.confirmation, key="admin-confirm-replace")

        if not st.button(
            "Import to database",
            type="primary",
            disabled=not plan.may_proceed(confirmed),
            key="admin-import-btn",
        ):
            return
        if not plan.may_proceed(confirmed):
            return

        try:
            stats = import_dataframe(df, id_column=id_column, table_name=table)
        except duckdb.Error as exc:
            logger.exception("import failed")
            st.error(_friendly_db_error(exc))
            return
        except Exception as exc:
            logger.exception("import failed")
            st.error(f"The import failed and nothing was changed: {exc}")
            return

        # No st.rerun() here: setting _IMPORTED is enough for the caller to
        # render step 2 in this same run, and a rerun would throw away the
        # success message the uploader needs to see.
        # The plan said "create" and that is no longer true.
        _forget_plan()
        st.session_state[_IMPORTED] = stats
        st.success(
            f"Imported {stats['rows']:,} rows × {stats['cols']:,} columns into "
            f"`{stats['table']}`."
        )
