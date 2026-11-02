"""Streamlit entry — Vespa-backed tips search."""

import streamlit as st

from re_source.brand import apply_branding
from re_source.nav import pages

st.set_page_config(page_title="Re-Source", layout="wide")

# Before any page renders, so every page — and every page-level stylesheet —
# sits on the same tokens.
apply_branding()

st.navigation(pages(), position="hidden").run()
