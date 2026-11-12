"""Streamlit render for the Vespa search UI."""

import html
import re
import time
import uuid
from datetime import datetime

import streamlit as st
from langfuse import propagate_attributes

from re_source import audit
from re_source.brand import STACKS as BRAND_STACKS
from re_source import config
from re_source.config import MAX_QUERY_CHARS
from re_source.disclosure import prepare_export
from re_source.export_selection import ExportSelection
from re_source import result_filter
from re_source.result_card import NOTHING_TO_SAY, Match, card_for
from re_source.search_runner import start_run
from re_source.search_session import SearchSession
from re_source.vespa_strategies import VESPA_STRATEGIES


APP_NAME = "Re-Source"


# ----- user identity + Langfuse attribution --------------------------------

def _user_id_from(user) -> str:
    """Resolve an identity from a dict-like st.user, or 'anonymous'."""
    for claim in ("email", "preferred_username", "sub"):
        try:
            value = user.get(claim)
        except Exception:
            return "anonymous"
        if value and str(value).strip():
            return str(value).strip()
    return "anonymous"


def current_user_id() -> str:
    """Current user's email for trace attribution, or 'anonymous'."""
    return _user_id_from(st.user)


def session_id() -> str:
    """Stable per-Streamlit-session id so a user's queries group in Langfuse."""
    return _session().trace_session_id(lambda: uuid.uuid4().hex)


def _audit_viewed(results, strategy: str, *, new_search: bool) -> None:
    """Record which records this user was shown."""
    session = _session()
    page = session.begin_audit_page(new_search=new_search)
    refs = session.unseen(audit.record_refs(results))
    if not refs:
        return
    audit.log_event(
        audit.EVENT_RESULTS_VIEWED,
        actor=current_user_id(),
        record_ids=refs,
        session_id=session_id(),
        surface=audit.SURFACE_STREAMLIT,
        strategy=strategy,
        page=page,
    )


def _auth_configured() -> bool:
    """True if an [auth] block exists in secrets — i.e. st.login can work."""
    try:
        return "auth" in st.secrets
    except Exception:
        return False


def _enforce_login() -> None:
    """Stop the app when REQUIRE_LOGIN is set but nobody's signed in."""
    if not config.REQUIRE_LOGIN:
        return
    if not _auth_configured():
        st.error("REQUIRE_LOGIN is set but .streamlit/secrets.toml has no [auth].")
        st.stop()
    if not getattr(st.user, "is_logged_in", False):
        st.info("Please log in to search.")
        # Signed-out users do not reach the header menu.
        st.button("Log in", type="primary", on_click=st.login, key="gate-login")
        st.stop()


def _render_app_menu(label: str) -> None:
    """The app's chrome — page switcher and sign in/out — behind the avatar."""
    from re_source.nav import pages

    # A help tooltip changes the popover trigger's layout dimensions.
    with st.popover(label):
        for page in pages():
            st.page_link(page)
        st.divider()
        if not _auth_configured():
            st.caption("Running anonymously — no auth provider configured.")
        elif getattr(st.user, "is_logged_in", False):
            who = st.user.get("name") or st.user.get("email") or "user"
            st.caption(f"Signed in as **{who}**")
            st.button("Log out", on_click=st.logout)
        else:
            st.button("Log in", type="primary", on_click=st.login)


def _session(state=None) -> SearchSession:
    """The current search, over `state` or this session's own."""
    return SearchSession(st.session_state if state is None else state)


def _reset_filters(state=None) -> None:
    """Put the filters back to showing everything.

    Args:
        state: session mapping; defaults to st.session_state.
    """
    _session(state).clear_filters()


def _reset_search(state=None) -> None:
    """Drop the current search so the controls come back empty.

    Args:
        state: session mapping; defaults to st.session_state.
    """
    _session(state).clear()


def _go_home() -> None:
    """Clear the current search and navigate to the default card view."""
    from re_source.nav import home_page

    _reset_search()
    st.switch_page(home_page())


def _render_home_button() -> None:
    """Render the wordmark and navigate only after the widget callback phase."""
    if st.button(APP_NAME, key="go-home", type="tertiary"):
        _go_home()


_HEADER_CSS = """
<style>
  /* --- Masthead ------------------------------------------------------------ */
  /* Sticky positioning needs the outer layout wrapper. */
  [data-testid="stLayoutWrapper"]:has(> [class*="st-key-app-masthead"]) {
    position: sticky;
    top: 0;
    z-index: 5;
    background: var(--warm-white);
  }
  [class*="st-key-app-masthead"] {
    background: var(--warm-white);
    min-height: 44px;
    align-content: center;
  }

  /* Present the home button as the wordmark. */
  [class*="st-key-home-title"] .stButton > button,
  [class*="st-key-home-title"] .stButton > button:hover {
    padding: 0;
    border: none;
    background: none;
  }
  [class*="st-key-home-title"] .stButton > button p {
    font-family: """ + BRAND_STACKS["label"] + """;
    font-weight: 700;
    font-size: 25px;
    letter-spacing: -0.01em;
    line-height: 1;
    color: var(--blue-03);
  }
  [class*="st-key-home-title"] .stButton > button:hover p {color: var(--blue-04);}

  /* --- Who's signed in, and where you can go -------------------------------- */
  [class*="st-key-masthead-row"] {
    width: 100%;
    flex-wrap: nowrap;
    align-items: center;
  }
  [class*="st-key-app-avatar"] {
    display: flex;
    justify-content: flex-end;
    align-items: center;
    min-height: 44px;
    margin-left: auto;
  }
  [class*="st-key-app-avatar"] [data-testid="stPopover"] {
    display: flex;
    align-items: center;
    justify-content: flex-end;
  }
  [class*="st-key-app-avatar"] [data-testid="stPopover"] button {
    width: 30px;
    min-width: 30px;
    height: 30px;
    min-height: 30px;
    padding: 0;
    border-radius: 50%;
    background: var(--warm-black);
    border: 1px solid var(--warm-black);
    color: var(--warm-white);
    display: grid;
    place-items: center;
    position: relative;
  }
  [class*="st-key-app-avatar"] [data-testid="stPopover"] button:hover {
    background: var(--blue-03);
    border-color: var(--blue-03);
    color: var(--warm-white);
  }
  [class*="st-key-app-avatar"] [data-testid="stPopover"] button p {
    position: absolute;
    inset: 0;
    display: grid;
    place-items: center;
    text-align: center;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.04em;
    line-height: 1;
    margin: 0;
    pointer-events: none;
  }
  /* Center Streamlit's inner label wrapper. */
  [class*="st-key-app-avatar"] button > div {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0;
  }
  /* Remove the aria-hidden chevron and its layout wrapper. */
  [class*="st-key-app-avatar"] button div:has(> span > [data-testid="stIconMaterial"]) {
    display: none;
  }
  [class*="st-key-app-avatar"] button p {white-space: nowrap;}
</style>
"""


def _avatar_initials() -> str:
    """Monogram for the masthead avatar: initials from the signed-in user's
    name, their email's first two letters as a fallback, and "?" when nobody is
    signed in (which is every request when no auth provider is configured)."""
    try:
        name = (st.user.get("name") or "").strip()
        email = (st.user.get("email") or "").strip()
    except Exception:
        name, email = "", ""
    if name:
        parts = [p for p in name.split() if p]
        return (parts[0][0] + parts[-1][0]).upper() if len(parts) > 1 else parts[0][:2].upper()
    if email:
        return email[:2].upper()
    return "?"


def render_header(title: str = APP_NAME) -> None:
    """The masthead every page starts with: the wordmark on the left, then the chrome menu
    and the signed-in avatar on the right.
    """
    _enforce_login()
    st.markdown(_HEADER_CSS, unsafe_allow_html=True)
    with st.container(key="app-masthead"):
        with st.container(
            key="masthead-row",
            horizontal=True,
            horizontal_alignment="distribute",
            vertical_alignment="center",
        ):
            with st.container(key="home-title", width="content"):
                _render_home_button()
            with st.container(key="app-avatar", width="content"):
                _render_app_menu(_avatar_initials())
    if title != APP_NAME:
        st.title(title)


def _selection() -> ExportSelection:
    """The include-in-export choice, over this session's state."""
    return ExportSelection(st.session_state)


# Keys must match VESPA_STRATEGIES.
_STRATEGY_UI: dict[str, dict] = {
    "Content": {
        "tab": "Search by topic",
        "summary": "SEARCH BY TOPIC",
        "aside_title": "Best for",
        "aside_body": "Finding an idea when you don't know the exact words a "
                      "source used.",
        "label": "Describe what you're looking for",
        "placeholder": "recurring water outages in apartment buildings",
        "button": "Search by topic",
    },
    "BM25": {
        "tab": "Exact keyword match",
        "summary": "SEARCH BY KEYWORD",
        "aside_title": "Best for",
        "aside_body": "A name, company, case number, or distinctive phrase you "
                      "expect to appear verbatim, including an email address, "
                      "phone number, or identifier. Use \"quotes\" for an exact "
                      "phrase, -word to leave a word out, and OR in capitals "
                      "to match either word.",
        "label": "Words to match",
        "placeholder": "Northwind Health",
        "button": "Match keywords",
    },
    "Author Profile + LLM": {
        "tab": "Find sources",
        "summary": "SEARCH BY SOURCE",
        "aside_title": "Best for",
        "aside_body": "Finding who sent a submission rather than what it says. "
                      "Describe the person you want to reach.",
        "label": "Describe the person",
        "placeholder": "Small-business owners in Arizona",
        "button": "Find sources",
    },
}

_STRATEGY_UI_FALLBACK = {
    "summary": "SEARCH",
    "aside_title": "Best for",
    "aside_body": "",
    "label": "Query",
    "placeholder": "",
    "button": "Search",
}


def _strategy_meta(name: str) -> dict[str, str]:
    """Display copy for a strategy, falling back to its raw name."""
    return _STRATEGY_UI.get(name) or {**_STRATEGY_UI_FALLBACK, "tab": name, "button": f"Search: {name}"}


def _default_strategy() -> str:
    """The strategy selected before the user picks one — the first registry entry."""
    return next(iter(VESPA_STRATEGIES))


INTRO_LINE = "Search and filter thousands of rows of source material."


def _render_intro() -> None:
    """The home-state intro line. Silent once a search has run."""
    st.markdown(f'<p class="intro-line">{html.escape(INTRO_LINE)}</p>',
                unsafe_allow_html=True)


def _render_strategy_strip() -> str:
    """The pick-your-search-first strip. Returns the selected strategy name."""
    labels = {name: _strategy_meta(name)["tab"] for name in VESPA_STRATEGIES}
    # The segmented control is not rendered on the results screen.
    remembered = _session().remembered_strategy()
    strategy = st.segmented_control(
        "Search type",
        list(VESPA_STRATEGIES),
        format_func=lambda name: labels[name],
        default=remembered if remembered in VESPA_STRATEGIES else _default_strategy(),
        required=True,
        key="search_strategy",
        label_visibility="collapsed",
        on_change=_reset_search,
    ) or _default_strategy()
    _session().remember_strategy(strategy)
    return strategy


def _render_help_aside(strategy: str) -> None:
    """The home-state help column: what the selected search strategy is best for."""
    meta = _strategy_meta(strategy)
    with st.container(key="help-aside"):
        st.markdown(
            f'<div class="aside-title">{html.escape(meta["aside_title"])}</div>'
            f'<p class="aside-body">{html.escape(meta["aside_body"])}</p>',
            unsafe_allow_html=True,
        )


def _render_query_form(strategy: str) -> tuple[str, bool]:
    """The query box for one strategy. Returns (query, submitted)."""
    meta = _strategy_meta(strategy)
    with st.container(key="home-search"):
        with st.form(f"search_form::{strategy}", border=False):
            query = st.text_input(
                meta["label"], key=f"q::{strategy}", placeholder=meta["placeholder"],
                label_visibility="collapsed",
                max_chars=MAX_QUERY_CHARS,
            )
            with st.container(
                key="search-actions",
                horizontal=True,
                vertical_alignment="center",
                gap="small",
            ):
                submitted = st.form_submit_button(meta["button"], type="primary")
                st.markdown(
                    '<span class="kbd-hint">or press <kbd>&#8629;</kbd></span>',
                    unsafe_allow_html=True,
                    width="content",
                )
    return query, submitted


# ----- shared search surface (controls / filters / export / load-more) ------

def _attributed(**metadata):
    """Stamp a Langfuse trace with who is searching."""
    return propagate_attributes(
        user_id=current_user_id(),
        session_id=session_id(),
        metadata={**metadata, "surface": "streamlit"},
    )


def _execute_search(strategy_name: str, query: str) -> None:
    """Run `query` through the named strategy and stash everything the results views read
    out of session_state.
    """
    progress = st.progress(0.08, text="Searching…")

    def on_stage(frac, text):
        progress.progress(frac, text=text)

    t0 = time.time()
    try:
        page = start_run(strategy_name, attribution=_attributed).first_page(
            query, on_stage=on_stage
        )
    except Exception as e:
        progress.empty()
        st.error(f"Search failed: {e}")
        st.stop()
    elapsed = time.time() - t0
    progress.empty()

    # Persisted so results and the Export button survive the download rerun.
    session = _session()
    session.start(results=page.results, metadata=page.metadata, elapsed=elapsed,
                  query=query, strategy=strategy_name,
                  pool=page.pool, cursor=page.cursor)
    session.clear_filters()

    _audit_viewed(page.results, strategy_name, new_search=True)


def _render_search_summary(strategy: str, query: str) -> None:
    """What you searched, on the page showing what it found."""
    meta = _strategy_meta(strategy)
    with st.container(key="search-summary"):
        icon, copy, action = st.columns(
            [0.5, 4, 1.2], gap="medium", vertical_alignment="center"
        )
        with icon:
            st.markdown(
                """
                <div class="summary-icon" aria-hidden="true">
                  <svg viewBox="0 0 24 24">
                    <circle cx="10.5" cy="10.5" r="6.5"></circle>
                    <path d="m15.4 15.4 4.1 4.1"></path>
                    <path d="m7.6 10.5 1.8 1.8 3.8-4"></path>
                  </svg>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with copy:
            st.markdown(
                f'<div class="summary-label">{html.escape(meta["summary"])}</div>'
                f'<div class="summary-query">“{html.escape(query)}”</div>',
                unsafe_allow_html=True,
            )
        with action:
            with st.container(key="summary-action"):
                st.button("Start over", icon=":material/replay:",
                          key="start-over", type="secondary",
                          on_click=_reset_search, use_container_width=True)


_SEARCH_SURFACE_CSS = """
<style>
  /* --- Intro line ---------------------------------------------------------- */
  .intro-line {
    font-family: """ + BRAND_STACKS["article"] + """;
    font-size: 19px;
    line-height: 1.45;
    color: var(--gray-04);
    max-width: 620px;
    margin: 0;
    padding: 0;
  }

  /* --- Strategy tabs ------------------------------------------------------- */
  [class*="st-key-strategy-strip"] {
    margin-top: 12px;
    margin-bottom: 0;
  }
  [class*="st-key-strategy-strip"] [data-testid="stButtonGroup"] {
    display: flex;
    justify-content: center;
    width: 100%;
    background: transparent;
    border: none;
    padding: 0;
  }
  [class*="st-key-strategy-strip"] [data-testid="stButtonGroup"] > div {
    display: flex;
    flex-direction: row;
    flex-wrap: nowrap;
    align-items: center;
    justify-content: center;
    gap: 4px;
    width: fit-content;
    max-width: 100%;
    background: var(--gray-01);
    border: none;
    border-radius: 999px;
    padding: 5px;
  }
  [class*="st-key-strategy-strip"] button[kind="segmented_control"]:not(:disabled),
  [class*="st-key-strategy-strip"] button[kind="segmented_controlActive"]:not(:disabled) {
    flex: 0 0 auto;
    width: auto;
    min-width: 0;
    height: auto;
    min-height: 0;
    margin: 0;
    padding: 12px 22px;
    border: none;
    border-radius: 999px;
  }
  [class*="st-key-strategy-strip"] button p {
    font-size: 14px;
    font-weight: 600;
    letter-spacing: -0.005em;
    line-height: 1;
    white-space: nowrap;
  }
  [class*="st-key-strategy-strip"] button[kind="segmented_control"]:not(:disabled) {
    background: transparent;
    color: var(--warm-black);
  }
  [class*="st-key-strategy-strip"] button[kind="segmented_control"]:not(:disabled):hover {
    background: var(--gray-02);
    color: var(--warm-black);
  }
  [class*="st-key-strategy-strip"] button[kind="segmented_controlActive"]:not(:disabled),
  [class*="st-key-strategy-strip"] button[kind="segmented_controlActive"]:not(:disabled):hover {
    background: var(--warm-black);
    color: var(--warm-white);
  }

  /* --- Home query field ---------------------------------------------------- */
  [class*="st-key-home-search"] [data-baseweb="input"] {
    background: var(--white);
    border: 1px solid var(--gray-02);
    border-radius: 6px;
  }
  [class*="st-key-home-search"] [data-baseweb="base-input"] {
    background: transparent;
    border: none;
  }
  [class*="st-key-home-search"] [data-baseweb="input"]:focus-within {
    outline: 2px solid var(--blue-03);
    outline-offset: 2px;
  }
  [class*="st-key-home-search"] [data-testid="InputInstructions"] {display: none;}

  [class*="st-key-home-search"] [data-testid="stFormSubmitButton"] button {
    background: var(--warm-black);
    color: var(--warm-white);
    border-color: var(--warm-black);
  }
  [class*="st-key-home-search"] [data-testid="stFormSubmitButton"] button:hover {
    background: var(--blue-03);
    border-color: var(--blue-03);
    color: var(--warm-white);
  }
  .kbd-hint {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    font-size: 12px;
    color: var(--gray-03);
    white-space: nowrap;
  }
  .kbd-hint kbd {
    font-family: """ + BRAND_STACKS["body"] + """;
    font-size: 11px;
    line-height: 1;
    border: 1px solid var(--gray-02);
    border-bottom-width: 2px;
    padding: 2px 5px;
  }

  /* --- Help aside ---------------------------------------------------------- */
  [class*="st-key-help-aside"] {
    padding-left: 24px;
  }
  .aside-title {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--gray-04);
    margin-bottom: 14px;
  }
  .aside-body {
    font-size: 13px;
    line-height: 1.5;
    color: var(--gray-05);
    margin: 0 0 10px;
  }

  /* --- The search, once it has run ------------------------------------------ */
  [class*="st-key-search-summary"] {
    margin-top: 12px;
    padding: 20px 28px 38px;
    background: var(--white);
  }
  [class*="st-key-search-summary"] [data-testid="stHorizontalBlock"] {
    align-items: center;
    flex-wrap: nowrap;
  }
  .summary-icon {
    width: 56px;
    height: 56px;
    border-radius: 50%;
    background: var(--blue-01);
    color: var(--blue-03);
    display: grid;
    place-items: center;
  }
  .summary-icon svg {
    width: 26px;
    height: 26px;
    fill: none;
    stroke: currentColor;
    stroke-width: 1.8;
    stroke-linecap: round;
    stroke-linejoin: round;
  }
  .summary-label {
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.12em;
    line-height: 1;
    color: var(--gray-04);
  }
  .summary-query {
    font-size: 26px;
    line-height: 1.2;
    letter-spacing: -0.01em;
    color: var(--warm-black);
    margin-top: 10px;
  }
  [class*="st-key-summary-action"] {align-items: flex-end;}

  /* --- "Full row" ---------------------------------------------------------- */
  [class*="st-key-fullrow-"] button {
    background: transparent;
    border-color: var(--gray-02);
    color: var(--warm-black);
  }
  [class*="st-key-fullrow-"] button:hover {
    background: var(--warm-black);
    border-color: var(--warm-black);
    color: var(--warm-white);
  }

  /* --- Nothing to show ----------------------------------------------------- */
  .empty-state {
    font-size: 14px;
    line-height: 1.5;
    color: var(--gray-04);
    max-width: 52ch;
    padding: 8px 0 24px;
  }

  /* --- Results section ----------------------------------------------------- */
  [class*="st-key-results-section"] {
    margin-top: 8px;
  }

  /* --- Results header bar -------------------------------------------------- */
  [class*="st-key-results-header"] {
    margin-bottom: 0;
    padding-bottom: 14px;
    border-bottom: 1px solid var(--gray-02);
  }
  [class*="st-key-results-list"] {
    margin-top: 0;
    gap: 16px;
  }
  .result-count {
    font-size: 14px;
    font-weight: 600;
    letter-spacing: 0.01em;
    color: var(--warm-black);
  }
  .result-sort {font-size: 12px; color: var(--gray-04); margin-top: 3px;}
  [class*="st-key-results-header"] [data-testid="stExpander"],
  [class*="st-key-results-header"] details {
    border: none;
    box-shadow: none;
    background: transparent;
    align-self: flex-start;
    width: auto;
  }
  [class*="st-key-results-header"] details summary {
    padding: 0.1rem 0;
    margin-left: 0;
    font-size: 12px;
    color: var(--gray-04);
  }
  [class*="st-key-export-action"] {align-items: flex-end;}
  [class*="st-key-results-header"] [data-testid="stDownloadButton"] button {
    background: var(--blue-03);
    color: var(--white);
    border-color: var(--blue-03);
  }
  [class*="st-key-results-header"] [data-testid="stDownloadButton"] button:disabled {
    background: transparent;
    border-color: var(--gray-02);
    color: var(--gray-03);
    cursor: not-allowed;
  }

  /* --- Filter panel -------------------------------------------------------- */
  [data-testid="stLayoutWrapper"]:has(> [class*="st-key-filter-rail"]) {
    position: sticky;
    top: 4.25rem;
  }
  [class*="st-key-filter-rail"] {
    background: var(--white);
    border: 1px solid var(--gray-02);
    border-radius: 10px;
    padding: 12px 20px 8px;
    gap: 8px;
  }
  [class*="st-key-filter-rail"] [data-testid="stMarkdownContainer"]:has(.filters-title) {
    margin-bottom: 12px;
  }
  .filters-title {
    font-family: """ + BRAND_STACKS["label"] + """;
    font-size: 19px;
    font-weight: 700;
    letter-spacing: -0.01em;
    line-height: 1;
    margin: 0;
    color: var(--warm-black);
  }
  [class*="st-key-filter-rail"] label p {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--gray-04);
  }
  [class*="st-key-filter-rail"] [data-baseweb="tag"] {
    background: var(--blue-01);
    color: var(--blue-04);
    border-radius: 4px;
    font-size: 12px;
    font-weight: 600;
  }
  [class*="st-key-filter-rail"] [data-baseweb="tag"] span,
  [class*="st-key-filter-rail"] [data-baseweb="tag"] svg {color: var(--blue-04);}
  [class*="st-key-filter-rail"] [data-baseweb="select"] > div,
  [class*="st-key-filter-rail"] [data-baseweb="input"] {
    background: var(--white);
    border: 1px solid var(--gray-02);
    border-radius: 6px;
    font-size: 13px;
  }
  [class*="st-key-filters-reset"] {
    margin-top: 6px;
    align-items: flex-start;
  }
</style>
"""


def _render_search_controls() -> None:
    """The masthead, and then either the search or a record of it: tabs plus the query
    field on the home state, and once results exist, a static line naming the strategy that
    ran and what was asked.
    """
    st.markdown(_SEARCH_SURFACE_CSS, unsafe_allow_html=True)
    render_header()

    active = _active_search()
    if active is not None:
        _render_search_summary(active.strategy, active.query)
        return

    _render_intro()
    with st.container(key="strategy-strip"):
        strategy = _render_strategy_strip()
    controls, aside = st.columns([1, 0.36], gap="large")
    with aside:
        _render_help_aside(strategy)
    with controls:
        query, submitted = _render_query_form(strategy)
        if submitted and query.strip():
            _execute_search(strategy, query.strip())
            st.rerun()


def _active_search():
    """The search on screen as an ActiveSearch, or None if nothing's run."""
    return _session().active()


def _render_filters(results):
    """Render the group, dataset and submitted-date filters in the current
    container and return the narrowed result list.
    """
    st.markdown('<div class="filters-title">Filters</div>', unsafe_allow_html=True)
    # Groups first, so the dataset menu offers only what they left showing.
    hidden = [
        exclusion.group for exclusion in result_filter.group_exclusions()
        if st.checkbox(exclusion.label, key=exclusion.key)
    ]
    results = result_filter.without_groups(results, hidden)
    on_offer = result_filter.reconcile(st.session_state, results)

    selected_datasets = st.multiselect(
        "Datasets", on_offer.datasets, default=on_offer.datasets,
        key=result_filter.DATASETS_KEY,
        help="Remove a dataset to drop its results from the view and the export.",
    )
    date_range = st.date_input(
        "Submitted between", value=(), key=result_filter.DATES_KEY,
        help="When the tip was submitted. Rows with no submitted date are "
             "excluded when a range is set.",
    )
    st.button("Reset filters", key="filters-reset", type="tertiary",
              on_click=_reset_filters)

    chosen = result_filter.date_choice(date_range)
    if chosen.incomplete:
        st.caption("Pick an end date to apply the range.")
    narrowed = result_filter.apply(
        results,
        result_filter.Narrowing(datasets=tuple(selected_datasets), dates=chosen.dates),
    )
    if narrowed.undated:
        st.caption(
            f"{narrowed.undated} with no submitted date are excluded by the "
            "date range."
        )
    return narrowed.results


def _result_count_label(filtered, results) -> str:
    """The number of responses currently visible."""
    shown = str(len(filtered))
    noun = "response" if len(filtered) == 1 else "responses"
    return f"{shown} {noun}"


def _result_sort_label(elapsed, used_strategy) -> str:
    """The line under the count: how the list is ordered and how long it took."""
    elapsed_str = f"{elapsed:.2f}s" if elapsed is not None else "-"
    return f"Sorted by relevance · {elapsed_str}"


def _render_export_button(export_rows, used_strategy) -> None:
    """Excel download for the currently-selected rows, in the current container."""
    if not export_rows:
        st.download_button("Export 0 selected", data=b"", file_name="results.xlsx",
                           disabled=True)
        return
    disclosure = prepare_export(
        export_rows,
        actor=current_user_id(),
        surface=audit.SURFACE_STREAMLIT,
        strategy=used_strategy,
        session_id=session_id(),
    )
    if st.download_button(
        f"Export {len(export_rows)} selected",
        data=disclosure.data,
        file_name=disclosure.filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ):
        disclosure.record()


def _render_results_header(filtered, results, elapsed, used_strategy, metadata,
                           export_rows) -> None:
    """The bar over the result list: what you got and how it's ordered on the
    left, what to do with what you've picked on the right.
    """
    with st.container(key="results-header"):
        summary, actions = st.columns([2, 1.5], vertical_alignment="center")
        with summary:
            st.markdown(
                f'<div class="result-count">{html.escape(_result_count_label(filtered, results))}</div>'
                f'<div class="result-sort">{html.escape(_result_sort_label(elapsed, used_strategy))}</div>',
                unsafe_allow_html=True,
            )
            if metadata:
                with st.expander("Search details"):
                    st.json(metadata)
        with actions:
            with st.container(key="export-action"):
                _render_export_button(export_rows, used_strategy)


_LOAD_MORE_CSS = """
<style>
  [class*="st-key-load-more"] {
    margin-top: 32px;
    align-items: center;
    gap: 0.5rem;
  }
  [class*="st-key-loadmore-button"] button {
    background: transparent;
    border-color: var(--warm-black);
    color: var(--warm-black);
  }
  [class*="st-key-loadmore-button"] button:hover {
    background: var(--warm-black);
    border-color: var(--warm-black);
    color: var(--warm-white);
  }
</style>
"""


def _render_empty_state(results, filtered) -> None:
    """Say so when the filters hide every result."""
    if filtered or not results:
        return
    st.markdown(
        '<div class="empty-state">No responses match the filters. '
        f'Reset them to see all {len(results)}.</div>',
        unsafe_allow_html=True,
    )


def _render_load_more() -> None:
    """Validate the next slice of the already-retrieved pool."""
    session = _session()
    active = session.active()
    if active is None:
        return
    if active.pool is None:
        return
    run = start_run(active.strategy, attribution=_attributed)
    plan = session.page_plan(run.page_size)
    if plan is None:
        return

    st.markdown(_LOAD_MORE_CSS, unsafe_allow_html=True)
    with st.container(key="load-more"):
        st.caption(f"Checks {plan.step} more of {plan.remaining} remaining.")
        clicked = st.button(f"Load {plan.step} more", key="loadmore-button",
                            type="secondary")
        if clicked:
            progress = st.progress(0.4, text="Checking the next batch…")
            _apply_load_more(session, run, active, plan)
            progress.progress(1.0, text="Done")
            progress.empty()
            st.rerun()


def _apply_load_more(session, run, active, plan) -> None:
    """Validate the planned slice, append it, and record only what's new."""
    more = run.next_page(active.query, plan.pool_slice)
    session.extend(more, plan.step)
    _audit_viewed(more, active.strategy, new_search=False)


def _highlight_html(text: str, highlights) -> str:
    """HTML-escape `text`, then wrap any of the verbatim `highlights` phrases
    (case-insensitive) in <mark>.

    Returns:
        A safe HTML string for st.markdown(..., unsafe_allow_html=True).
    """
    escaped = html.escape(text or "")
    phrases = sorted(
        {p.strip() for p in (highlights or []) if p and p.strip()},
        key=len, reverse=True,
    )
    if not phrases:
        return escaped
    pattern = re.compile(
        "|".join(re.escape(html.escape(p)) for p in phrases), re.IGNORECASE
    )
    return pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", escaped)


def _excerpt(text: str, highlights, radius: int = 140):
    """Return (prefix, body, suffix): a window of `text` centered on the first highlight
    phrase (case-insensitive), or a head window when there's no highlight match.

    Returns:
        (prefix, body, suffix). `prefix`/`suffix` are "…" when the window is
        clipped at that end, else "".
    """
    text = text or ""
    pos = -1
    for h in highlights or []:
        h = (h or "").strip()
        if not h:
            continue
        i = text.lower().find(h.lower())
        if i != -1 and (pos == -1 or i < pos):
            pos = i
    if pos == -1:
        start, end = 0, min(len(text), radius * 2)
    else:
        start = max(0, pos - radius)
        end = min(len(text), pos + radius)
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(text) else ""
    return prefix, text[start:end], suffix


def _card_meta_html(card) -> str:
    """The card's top line: row number, the dataset it came from as a tag, and when
    it was submitted.
    """
    parts = [
        f'<span class="card-num">{card.index:02d}</span>',
        f'<span class="dataset-tag">{html.escape(card.dataset)}</span>',
    ]
    if card.submitted:
        parts.append(
            '<span class="card-meta">'
            f'submitted {html.escape(card.submitted.isoformat())}</span>'
        )
    return f'<div class="card-head">{"".join(parts)}</div>'


def _field_label(label: str) -> str:
    label = label.strip()
    return f"{label[:1].upper()}{label[1:]}" if label else ""


_RESULT_CARD_CSS = """
<style>
  [class*="st-key-scard-"] {
    background: var(--white);
    border: 1px solid var(--card-border);
    border-radius: 10px;
    box-shadow: none;
    padding: 20px 24px;
  }
  [class*="st-key-scard-"]:hover,
  [class*="st-key-scard-"]:focus-within {
    border-color: var(--blue-03);
  }
  [class*="st-key-scard-inc-"] {
    border-color: var(--blue-03);
    box-shadow: inset 0 0 0 1px var(--blue-03),
                0 4px 16px rgba(17, 17, 16, 0.13);
  }
  [class*="st-key-scard-"] {gap: 8px;}

  .card-head {display: flex; align-items: center; gap: 10px; flex-wrap: wrap;}
  .card-num {
    font-family: """ + BRAND_STACKS["article"] + """;
    font-size: 13px;
    color: var(--gray-03);
  }
  .dataset-tag {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--gray-04);
  }
  .card-meta {font-size: 12px; color: var(--gray-04); white-space: nowrap;}

  .card-match + .card-match {margin-top: 26px;}
  .card-question {
    font-size: 13px;
    line-height: 1.45;
    color: var(--gray-04);
    max-width: 72ch;
    margin-bottom: 4px;
  }

  .tip-quote {
    font-family: """ + BRAND_STACKS["article"] + """;
    font-size: 16px;
    line-height: 1.5;
    letter-spacing: -0.005em;
    color: var(--warm-black);
    max-width: 60ch;
    margin: 0;
  }
  .tip-quote mark {padding: 0;}
  [class*="st-key-cardfoot-"] {
    margin-top: 16px;
    padding: 0;
    border: none;
    background: transparent;
  }
  [class*="st-key-cardpick-"],
  [class*="st-key-cardopen-"] {align-items: flex-end;}
  .why-label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--gray-04);
    margin-bottom: 4px;
  }
  .why-text {
    font-size: 14px;
    line-height: 1.45;
    color: var(--gray-05);
    max-width: 62ch;
  }
</style>
"""


def _toggle_export(r) -> None:
    """Flip a result's include-in-export choice — the callback both views
    hand to their control."""
    _selection().toggle(r)


@st.dialog("Full row", width="large")
def _show_full_row(fields: dict, url: str | None = None) -> None:
    """Modal showing a result's full row.

    Args:
        url: when given, an 'Open original record' link is shown above the row.
    """
    if url:
        st.markdown(f"[↗ Open original record]({url})")
    st.json(fields)


_COLUMNS_CARD_CSS = """
            <style>
            [class*="st-key-card-inc-"] {
                border: 2px solid var(--blue-03) !important;
                border-radius: 10px;
                background: var(--white);
            }
            [class*="st-key-card-exc-"] {
                border: 1px dashed var(--gray-03) !important;
                border-radius: 10px;
                opacity: 0.55;
            }
            </style>
            """


def _render_result_card(i, r, sel) -> None:
    """One result as a bordered card: where it came from, the fields it
    matched in as question-over-answer, and why the LLM kept it."""
    card = card_for(i, r, sel)

    with st.container(border=True,
                      key=f"scard-{'inc' if card.included else 'exc'}-{card.ident}"):
        h_left, h_right = st.columns([4, 1], vertical_alignment="center")
        with h_left:
            st.markdown(_card_meta_html(card), unsafe_allow_html=True)
        with h_right:
            with st.container(key=f"cardpick-{card.ident}"):
                # Include state in the key so Streamlit remounts the checkbox.
                st.checkbox(
                    "Export",
                    key=f"cardcheck-{'inc' if card.included else 'exc'}-{card.ident}",
                    value=card.included,
                    on_change=_toggle_export,
                    args=(r,),
                )

        matches = card.matches or (Match("", card.fallback),)
        blocks = []
        for match in matches:
            if not match.answer:
                continue
            pfx, mid, sfx = _excerpt(match.answer, card.highlights)
            question = (f'<div class="card-question">'
                        f'{html.escape(_field_label(match.question))}</div>'
                        if match.question else "")
            blocks.append(
                f'<div class="card-match">{question}'
                f'<div class="tip-quote">“{pfx}{_highlight_html(mid, card.highlights)}{sfx}”</div>'
                f'</div>'
            )
        if blocks:
            st.markdown("".join(blocks), unsafe_allow_html=True)

        with st.container(key=f"cardfoot-{card.ident}"):
            f_left, f_right = st.columns([3, 1], vertical_alignment="bottom")
            with f_left:
                st.markdown(
                    '<div class="why-label">Why this matched</div>'
                    f'<div class="why-text">{html.escape(card.why)}</div>',
                    unsafe_allow_html=True,
                )
            with f_right:
                with st.container(key=f"cardopen-{card.ident}"):
                    if card.fields and st.button(
                        "Full row", key=f"fullrow-{card.ident}", type="secondary"
                    ):
                        _show_full_row(card.fields, card.url)


def _render_result_columns(i, r, sel) -> None:
    """One result as a roomy card with its content columnized: the include
    toggle and metadata, the matching evidence, and the LLM side by side."""
    card = card_for(i, r, sel)

    with st.container(border=True,
                      key=f"card-{'inc' if card.included else 'exc'}-{card.ident}"):
        st.button(
            "✓ Included — click to drop" if card.included else "Excluded — click to include",
            key=f"toggle-{card.ident}",
            use_container_width=True,
            type="primary" if card.included else "secondary",
            on_click=_toggle_export,
            args=(r,),
        )
        st.markdown(f"### {card.dataset}")
        meta_c, evidence_c, llm_c = st.columns([1.2, 2, 2])

        with meta_c:
            st.markdown(f"**#{card.index}**")
            if card.matched_fields != NOTHING_TO_SAY:
                st.caption(f"Fields: {card.matched_fields}")
            if card.submitted:
                st.caption(f"Submitted {card.submitted.isoformat()}")
            if card.url:
                st.markdown(f"[↗ Open original record]({card.url})")
            if card.fields and st.button("Full row", key=f"fullrow-{card.ident}"):
                _show_full_row(card.fields, card.url)

        with evidence_c:
            st.markdown(f"**{card.evidence_label}**")
            if card.matches:
                for match in card.matches:
                    with st.container(border=True):
                        if match.question:
                            st.caption(match.question)
                        st.markdown(
                            _highlight_html(match.answer, card.highlights),
                            unsafe_allow_html=True,
                        )
            else:
                st.write(NOTHING_TO_SAY)

        with llm_c:
            st.markdown("**LLM**")
            st.write(card.why)


# ----- the results page -----------------------------------------------------


def _render_results_page(*, result_css: str, render_result) -> None:
    """The results page minus the results."""
    _render_search_controls()

    active = _active_search()
    if active is None:
        return
    results = active.results

    with st.container(key="results-section"):
        rail, main = st.columns([1, 3.2], gap="large", vertical_alignment="top")
    with rail:
        with st.container(border=True, key="filter-rail"):
            filtered = _render_filters(results)

    with main:
        # Emit CSS before the visible stack so it does not add an internal gap.
        st.markdown(result_css, unsafe_allow_html=True)

        selection = _selection()
        export_rows = selection.rows(filtered)
        _render_results_header(filtered, results, active.elapsed, active.strategy,
                               active.metadata, export_rows)

        with st.container(key="results-list"):
            _render_empty_state(results, filtered)

            for i, r in enumerate(filtered, 1):
                render_result(i, r, selection)

        _render_load_more()


# ----- pages ----------------------------------------------------------------


def render_vespa_search() -> None:
    """Card view (default page)."""
    _render_results_page(result_css=_RESULT_CARD_CSS,
                         render_result=_render_result_card)


def render_vespa_table() -> None:
    """Columns view (prototype alternative UI)."""
    _render_results_page(result_css=_COLUMNS_CARD_CSS,
                         render_result=_render_result_columns)
