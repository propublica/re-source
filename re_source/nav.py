"""The app's page list, in one place."""

import streamlit as st


def home_page():
    """Build the default page so other controls can navigate home explicitly."""
    from re_source.search_ui import render_vespa_search

    return st.Page(render_vespa_search, title="Card view", icon="🗂", default=True)


def pages() -> list:
    """The navigation entries, in menu order."""
    from re_source.admin_ui import render_admin_upload
    from re_source.search_ui import render_vespa_table

    return [
        home_page(),
        st.Page(render_vespa_table, title="Columns view", icon="📊"),
        st.Page(render_admin_upload, title="Upload data", icon="⬆"),
    ]
