"""Design system for the Streamlit surfaces."""

from __future__ import annotations

import streamlit as st


# ----- brand colors ---------------------------------------------------------

COLORS = {
    # Core neutrals
    "warm_black": "#111110",
    "warm_white": "#F2F1ED",

    # Functional neutrals
    "white": "#FFFFFF",
    "gray_01": "#E3E2DD",
    "gray_02": "#C8C6C0",
    "gray_03": "#92908B",
    "gray_04": "#5F5E59",
    "gray_05": "#31302D",
    "card_border": "#E1DED6",
    "black": "#000000",

    # Highlights
    "blue_01": "#BCD9ED",
    "blue_02": "#52A7E0",
    "blue_03": "#006EB7",
    "blue_04": "#015086",
}


# ----- typography -----------------------------------------------------------

# Native UI fonts keep the public and private apps visually identical without
# shipping or loading separately licensed font files.
SYSTEM_FONT_STACK = (
    'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif'
)
STACKS = {
    role: SYSTEM_FONT_STACK
    for role in ("display", "headline", "label", "body", "article")
}


def _tokens_css() -> str:
    """The palette as CSS custom properties."""
    lines = "\n".join(
        f"        --{name.replace('_', '-')}: {value};"
        for name, value in COLORS.items()
    )
    return f":root {{\n{lines}\n    }}"


def brand_css() -> str:
    """The full stylesheet, as a <style> block."""
    return f"""
<style>
/* ===== DESIGN TOKENS ===================================================== */

{_tokens_css()}


/* ===== APP =============================================================== */

html,
body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {{
    background-color: var(--warm-white);
    color: var(--warm-black);
}}

/* Streamlit's own toolbar — Deploy, the hamburger, the running indicator —
   sits over the top 60px of every page. This app has its own masthead there,
   and a second chrome bar above it is both noise and 60px the page can't use.
   Progress during a search is reported by st.progress in the page body, so
   nothing here is the only signal for anything. */
[data-testid="stHeader"] {{
    display: none;
}}

.block-container {{
    max-width: 1200px;
    /* The masthead is the first thing on the page and belongs at the top of it;
       2rem of dead space above the wordmark just pushed everything down. */
    padding-top: 0.5rem;
    padding-bottom: 4rem;
}}


/* ===== TYPOGRAPHY ======================================================== */

/* Display headline — medium, -1.5% tracking, 95% leading. */
.display {{
    font-family: {STACKS["display"]};
    font-weight: 500;
    letter-spacing: -0.015em;
    line-height: 0.95;
    color: var(--warm-black);
}}

/* Headline — medium, -0.5% tracking, 100% leading. */
.headline {{
    font-family: {STACKS["headline"]};
    font-weight: 500;
    letter-spacing: -0.005em;
    line-height: 1;
    color: var(--warm-black);
}}

/* Label — bold, +1% tracking, 100% leading. */
.label {{
    font-family: {STACKS["label"]};
    font-weight: 700;
    letter-spacing: 0.01em;
    line-height: 1;
    color: var(--warm-black);
}}

/* Subhead — medium, -1% tracking, 120% leading. */
.subhead {{
    font-family: {STACKS["body"]};
    font-weight: 500;
    letter-spacing: -0.01em;
    line-height: 1.2;
    color: var(--warm-black);
}}

/* Body — regular, 0% tracking, 135% leading. */
.body {{
    font-family: {STACKS["body"]};
    font-weight: 400;
    letter-spacing: 0;
    line-height: 1.35;
    color: var(--warm-black);
}}

/* Article body — medium, 0% tracking, 145% leading. */
.article-body {{
    font-family: {STACKS["article"]};
    font-weight: 500;
    letter-spacing: 0;
    line-height: 1.45;
    color: var(--warm-black);
}}

/* CTA + functional text — medium, +1% tracking, 130% leading. */
.functional {{
    font-family: {STACKS["body"]};
    font-weight: 500;
    letter-spacing: 0.01em;
    line-height: 1.3;
}}


/* ===== STREAMLIT DEFAULT TYPOGRAPHY ====================================== */

html,
body,
p,
li,
label,
input,
textarea,
button,
[data-testid="stMarkdownContainer"] {{
    font-family: {STACKS["body"]};
}}

/* Streamlit h1 = display. */
h1 {{
    font-family: {STACKS["display"]};
    font-weight: 500;
    letter-spacing: -0.015em;
    line-height: 0.95;
    color: var(--warm-black);
}}

/* Streamlit h2 / h3 = headline. */
h2,
h3 {{
    font-family: {STACKS["headline"]};
    font-weight: 500;
    letter-spacing: -0.005em;
    line-height: 1;
    color: var(--warm-black);
}}


/* ===== LINKS ============================================================= */

a {{
    color: var(--blue-03);
    text-decoration-color: var(--blue-03);
}}

a:hover {{
    color: var(--blue-04);
    text-decoration-color: var(--blue-04);
}}


/* ===== BUTTONS =========================================================== */

/* Keep Streamlit's native button geometry and typography. Only map its button
   states onto the brand palette. */

[class*="stBaseButton-primary"] {{
    background-color: var(--blue-03);
    color: var(--white);
    border-color: var(--blue-03);
}}

[class*="stBaseButton-primary"]:hover {{
    background-color: var(--blue-04);
    border-color: var(--blue-04);
    color: var(--white);
}}

[class*="stBaseButton-secondary"] {{
    background-color: var(--white);
    color: var(--blue-03);
    border-color: var(--gray-02);
}}

[class*="stBaseButton-secondary"]:hover {{
    background-color: var(--gray-01);
    border-color: var(--blue-03);
    color: var(--blue-04);
}}

[class*="stBaseButton-tertiary"] {{
    background-color: transparent;
    color: var(--blue-03);
    border-color: transparent;
}}

[class*="stBaseButton-tertiary"]:hover {{
    background-color: transparent;
    color: var(--blue-04);
}}

:focus-visible {{
    outline-color: var(--blue-03);
}}


/* ===== INPUTS ============================================================ */

/* base-input is the inner div baseweb paints Streamlit's own grey fill on. The
   outer wrappers are transparent, so setting only those leaves every input in
   the app looking stock. */
[data-baseweb="input"],
[data-baseweb="textarea"],
[data-baseweb="base-input"],
[data-baseweb="select"] > div {{
    background-color: var(--white);
}}

[data-baseweb="input"] input,
[data-baseweb="textarea"] textarea {{
    font-family: {STACKS["body"]};
}}


/* ===== HIGHLIGHTS ======================================================== */

/* Both result views mark the query-relevant span with <mark>; the browser's
   default is a yellow no palette here would allow. It lives in the design
   system rather than one view's stylesheet because both views use it. */

mark {{
    background-color: var(--blue-01);
    color: var(--warm-black);
    padding: 0 2px;
    border-radius: 0;
}}


/* ===== PROGRESS ========================================================== */

/* The search progress bar. Streamlit's own is its stock blue on a cool grey
   track, both of which sit oddly against a warm palette — and it's the most
   prominent thing on screen during the slow semantic search, so it's the last
   place to be showing another product's colours. The fill has no test id of its
   own; it's the track's only child. */

[data-testid="stProgressBarTrack"] {{
    background-color: var(--gray-01);
}}

[data-testid="stProgressBarTrack"] > div {{
    background-color: var(--blue-03);
}}


/* ===== CHECKBOXES ======================================================== */

/* Keep Streamlit's native checkbox size and shape; only map its states onto the
   brand palette. */

[data-testid="stCheckbox"] label > span:first-of-type {{
    border-color: var(--gray-03);
    background-color: var(--white);
}}

[data-testid="stCheckbox"] label:has(input:checked) > span:first-of-type {{
    border-color: var(--blue-03);
    background-color: var(--blue-03);
}}


/* ===== SIDEBAR =========================================================== */
/* This app has no sidebar (chrome is in the header menu, filters are a
   column), but the rule stays so the system drops into a sidebar app as-is. */

[data-testid="stSidebar"] {{
    background-color: var(--gray-01);
    border-right: 1px solid var(--gray-02);
}}

[data-testid="stSidebar"] * {{
    font-family: {STACKS["body"]};
}}


/* ===== DIVIDERS ========================================================== */

hr {{
    border: 0;
    border-top: 1px solid var(--gray-02);
}}


/* ===== UTILITY CLASSES =================================================== */

.text-blue {{color: var(--blue-03);}}
.text-muted {{color: var(--gray-04);}}
.blue-rule {{border-top: 2px solid var(--blue-02); padding-top: 1rem;}}
.brand-card {{background: var(--white); border: 1px solid var(--gray-02); padding: 1.25rem;}}
</style>
"""


def apply_branding() -> None:
    """Install the design system. Call once per script run, before any page."""
    st.markdown(brand_css(), unsafe_allow_html=True)
