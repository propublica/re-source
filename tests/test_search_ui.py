"""The shared search UI module imports cleanly and exposes the registry + render."""

import inspect
import re
from types import SimpleNamespace

import pytest

import re_source.search_ui as ui


def test_render_is_callable():
    assert callable(ui.render_vespa_search)


def test_avatar_and_initials_are_centered_in_the_masthead():
    css = ui._HEADER_CSS
    assert "align-items: center;" in css
    assert "[class*=\"st-key-app-avatar\"] button > div" in css
    assert "justify-content: center;" in css
    assert "margin: 0;" in css


def test_masthead_uses_one_full_width_row_for_wordmark_and_menu():
    source = inspect.getsource(ui.render_header)
    assert 'key="masthead-row"' in source
    assert "horizontal=True" in source
    assert 'horizontal_alignment="distribute"' in source
    assert "st.columns" not in source

    assert '[class*="st-key-masthead-row"]' in ui._HEADER_CSS
    assert "margin-left: auto;" in ui._HEADER_CSS


def test_avatar_text_is_centered_independently_of_the_popover_chevron():
    css = ui._HEADER_CSS
    rule = css.split(
        '[class*="st-key-app-avatar"] [data-testid="stPopover"] button p {', 1
    )[1].split("}", 1)[0]

    assert "position: absolute;" in rule
    assert "inset: 0;" in rule
    assert "display: grid;" in rule
    assert "place-items: center;" in rule
    assert "text-align: center;" in rule


def test_avatar_label_wrapper_fills_the_whole_button():
    css = ui._HEADER_CSS
    rule = css.split(
        '[class*="st-key-app-avatar"] button > div {', 1
    )[1].split("}", 1)[0]

    assert "position: absolute;" in rule
    assert "inset: 0;" in rule
    assert "width: 100%;" in rule
    assert "height: 100%;" in rule


def test_search_field_uses_streamlit_defaults_and_enter_hint_hugs_the_button():
    assert '[class*="st-key-home-search"] input {' not in ui._SEARCH_SURFACE_CSS
    source = inspect.getsource(ui._render_query_form)
    assert 'key="search-actions"' in source
    assert "horizontal=True" in source
    assert 'gap="small"' in source
    assert 'or press <kbd>&#8629;</kbd>' in source
    assert ".kbd-hint" in ui._SEARCH_SURFACE_CSS
    assert "display: inline-flex;" in ui._SEARCH_SURFACE_CSS


def test_result_cards_use_one_compact_spacing_system_without_a_rule():
    css = ui._RESULT_CARD_CSS

    assert "padding: 20px 24px;" in css
    assert '[class*="st-key-scard-"] {gap: 8px;}' in css
    foot = css.split('[class*="st-key-cardfoot-"] {', 1)[1].split("}", 1)[0]
    assert "margin-top: 16px;" in foot
    assert "padding: 0;" in css
    assert "border-top:" not in css

    question = css.split(".card-question {", 1)[1].split("}", 1)[0]
    why = css.rsplit(".why-label {", 1)[1].split("}", 1)[0]
    assert "margin-bottom: 4px;" in question
    assert "margin-bottom: 4px;" in why


def test_selected_result_cards_keep_the_blue_focus_treatment():
    css = ui._RESULT_CARD_CSS
    base = css.split('[class*="st-key-scard-"] {', 1)[1].split("}", 1)[0]

    assert "border: 1px solid var(--card-border);" in base
    assert "box-shadow: none;" in base
    assert '[class*="st-key-scard-"]:hover' in css
    assert '[class*="st-key-scard-"]:focus-within' in css
    assert "border-color: var(--blue-03);" in css
    selected = css.split('[class*="st-key-scard-inc-"] {', 1)[1].split("}", 1)[0]
    assert "border-color: var(--blue-03);" in selected
    assert "box-shadow:" in selected
    assert '[class*="st-key-scard-exc-"] {' not in css
    assert "opacity:" not in css


def test_result_field_labels_keep_the_original_body_typography():
    assert ui._field_label("response") == "Response"
    assert ui._field_label("email Address") == "Email Address"
    assert ui._field_label("  response  ") == "Response"

    question = ui._RESULT_CARD_CSS.split(".card-question {", 1)[1].split("}", 1)[0]
    why = ui._RESULT_CARD_CSS.rsplit(".why-label {", 1)[1].split("}", 1)[0]
    assert "font-size: 13px;" in question
    assert "font-weight:" not in question
    assert "letter-spacing:" not in question
    assert "font-size: 11px;" in why
    assert "font-weight: 700;" in why
    assert "letter-spacing: 0.1em;" in why


def test_full_row_action_sits_in_the_card_footer():
    source = inspect.getsource(ui._render_result_card)

    assert "h_left, h_right = st.columns" in source
    assert "h_full" not in source
    assert source.index('key=f"fullrow-{card.ident}"') > source.index(
        'key=f"cardfoot-{card.ident}"'
    )
    footer = source.split('key=f"cardfoot-{card.ident}"', 1)[1]
    assert "f_left, f_right = st.columns" in footer


def test_home_strategy_selector_is_a_centered_nested_pill():
    css = ui._SEARCH_SURFACE_CSS
    section = css.split("/* --- Strategy tabs", 1)[1].split(
        "/* --- Home query field", 1
    )[0]

    assert "justify-content: center;" in section
    assert "width: 100%;" in section
    assert "width: fit-content;" in section
    assert "background: var(--gray-01);" in section
    assert "border-radius: 999px;" in section
    assert "background: var(--warm-black);" in section
    assert "color: var(--warm-white);" in section
    assert "border-bottom-color" not in section
    assert "box-shadow" not in section


class _TrackedRegion:
    def __init__(self, owner, name):
        self.owner = owner
        self.name = name
        self.previous = None

    def __enter__(self):
        self.previous = self.owner.current_region
        self.owner.current_region = self.name
        return self

    def __exit__(self, *_args):
        self.owner.current_region = self.previous


class _ProgressPlacementSt:
    def __init__(self):
        self.current_region = None
        self.progress_regions = []

    def markdown(self, *_args, **_kwargs):
        pass

    def container(self, key=None, **_kwargs):
        return _TrackedRegion(self, key)

    def columns(self, *_args, **_kwargs):
        return (_TrackedRegion(self, "controls"), _TrackedRegion(self, "aside"))

    def caption(self, *_args, **_kwargs):
        pass

    def button(self, *_args, **_kwargs):
        return True

    def progress(self, *_args, **_kwargs):
        self.progress_regions.append(self.current_region)
        return SimpleNamespace(progress=lambda *_a, **_kw: None, empty=lambda: None)

    def rerun(self):
        pass


def test_search_progress_stays_beneath_the_search_form(monkeypatch):
    fake = _ProgressPlacementSt()
    observed = []
    monkeypatch.setattr(ui, "st", fake)
    monkeypatch.setattr(ui, "render_header", lambda: None)
    monkeypatch.setattr(ui, "_active_search", lambda: None)
    monkeypatch.setattr(ui, "_render_intro", lambda: None)
    monkeypatch.setattr(ui, "_render_strategy_strip", lambda: "Content")
    monkeypatch.setattr(ui, "_render_query_form", lambda _mode: ("query", True))
    monkeypatch.setattr(ui, "_render_help_aside", lambda _mode: None)
    monkeypatch.setattr(
        ui,
        "_execute_search",
        lambda *_args: observed.append(fake.current_region),
    )

    ui._render_search_controls()

    assert observed == ["controls"]


def test_load_more_progress_stays_with_its_button(monkeypatch):
    fake = _ProgressPlacementSt()
    active = SimpleNamespace(strategy="Content", query="q", pool=[object()])
    plan = SimpleNamespace(step=1, remaining=1, pool_slice=[object()])
    session = SimpleNamespace(active=lambda: active, page_plan=lambda _size: plan)
    run = SimpleNamespace(page_size=1)
    monkeypatch.setattr(ui, "st", fake)
    monkeypatch.setattr(ui, "_session", lambda: session)
    monkeypatch.setattr(ui, "start_run", lambda *_args, **_kwargs: run)
    monkeypatch.setattr(ui, "_apply_load_more", lambda *_args: None)

    ui._render_load_more()

    assert fake.progress_regions == ["load-more"]


def test_ordinary_action_buttons_leave_geometry_and_type_to_streamlit():
    css = ui._SEARCH_SURFACE_CSS + ui._LOAD_MORE_CSS
    ordinary_actions = (
        "stFormSubmitButton",
        "st-key-fullrow-",
        "stDownloadButton",
        "st-key-filters-reset",
        "st-key-loadmore-button",
    )
    forbidden = {
        "border",
        "border-radius",
        "border-top",
        "font-size",
        "font-weight",
        "min-height",
        "padding",
        "text-decoration",
        "text-underline-offset",
        "white-space",
        "width",
    }

    for selectors, body in re.findall(r"([^{}]+)\{([^{}]*)\}", css):
        if not any(marker in selectors for marker in ordinary_actions):
            continue
        properties = set(re.findall(r"(?m)^\s*([\w-]+)\s*:", body))
        assert not properties & forbidden, (selectors.strip(), properties & forbidden)


def test_registry_has_expected_strategies():
    assert set(ui.VESPA_STRATEGIES) == {"Content", "BM25", "Author Profile + LLM"}


def test_highlight_html_wraps_case_insensitively():
    out = ui._highlight_html("There was MOLD in the unit", ["mold"])
    assert "<mark>MOLD</mark>" in out


def test_highlight_html_escapes_before_marking():
    # HTML in the source text is neutralized; the highlight still wraps.
    out = ui._highlight_html("a <b> & mold here", ["mold"])
    assert "&lt;b&gt;" in out and "&amp;" in out
    assert "<mark>mold</mark>" in out


def test_highlight_html_no_highlights_is_escaped_plain():
    out = ui._highlight_html("a & b", [])
    assert out == "a &amp; b"
    assert "<mark>" not in out


def test_highlight_html_non_matching_phrase_is_noop():
    # A paraphrased highlight that isn't a verbatim substring adds no <mark>.
    out = ui._highlight_html("the ceiling was damp", ["water damage"])
    assert "<mark>" not in out


def test_result_quotes_hug_the_text_and_highlight_only_the_match():
    source = inspect.getsource(ui._render_result_card)

    assert 'class="tip-quote">“{pfx}' in source
    assert '{sfx}”</div>' in source
    assert ".tip-quote mark {padding: 0;}" in ui._RESULT_CARD_CSS


def test_excerpt_centers_on_highlight():
    text = "x" * 300 + " NEEDLE " + "y" * 300
    pfx, body, sfx = ui._excerpt(text, ["needle"], radius=50)
    assert "NEEDLE" in body
    assert pfx == "…" and sfx == "…"


def test_excerpt_head_window_when_no_highlight():
    pfx, body, sfx = ui._excerpt("a" * 500, [], radius=100)
    assert pfx == ""        # head window is not clipped at the start
    assert sfx == "…"       # but is clipped at the end
    assert len(body) == 200


def test_excerpt_short_text_has_no_ellipsis():
    pfx, body, sfx = ui._excerpt("short text", ["text"], radius=140)
    assert pfx == "" and sfx == ""
    assert body == "short text"


# ----- search strategies -----------------------------------------------------
#
# The strategy strip is driven by _STRATEGY_UI, which has to stay in step with the
# strategy registry — a strategy with no display copy would otherwise render a form
# with no heading, label, or button text.

_COPY_KEYS = {
    "tab", "summary", "aside_title", "aside_body", "label",
    "placeholder", "button"
}


def test_every_strategy_has_display_copy():
    assert set(ui._STRATEGY_UI) == set(ui.VESPA_STRATEGIES)


def test_every_strategy_carries_the_full_copy_set():
    for name, meta in ui._STRATEGY_UI.items():
        assert _COPY_KEYS <= set(meta), name
        # All of it renders unconditionally on the home state.
        assert all(meta[k] for k in _COPY_KEYS), name


def test_home_aside_has_one_heading_and_no_example_buttons():
    assert {meta["aside_title"] for meta in ui._STRATEGY_UI.values()} == {"Best for"}
    assert all(
        not meta["aside_body"].lower().startswith("best")
        for meta in ui._STRATEGY_UI.values()
    )
    source = inspect.getsource(ui._render_help_aside)
    assert "st.button" not in source
    assert "examples" not in source


def test_keyword_help_names_the_search_operators():
    body = ui._STRATEGY_UI["BM25"]["aside_body"]
    assert '"quotes"' in body and "-word" in body and "OR in capitals" in body


def test_placeholders_do_not_use_example_abbreviations():
    for name, meta in ui._STRATEGY_UI.items():
        assert "e.g." not in meta["placeholder"].lower(), name


def test_results_summary_names_the_kind_of_search():
    assert {
        name: meta["summary"] for name, meta in ui._STRATEGY_UI.items()
    } == {
        "Content": "SEARCH BY TOPIC",
        "BM25": "SEARCH BY KEYWORD",
        "Author Profile + LLM": "SEARCH BY SOURCE",
    }


def test_results_summary_matches_the_search_callout_layout():
    source = inspect.getsource(ui._render_search_summary)
    assert "summary-icon" in source
    assert 'meta["summary"]' in source
    assert 'st.button("Start over"' in source
    assert "on_click=_reset_search" in source

    section = ui._SEARCH_SURFACE_CSS.split(
        "/* --- The search, once it has run", 1
    )[1].split('/* --- "Full row"', 1)[0]
    assert "background: var(--white);" in section
    assert "border-left" not in section
    assert "padding: 20px 28px 38px;" in section
    assert "border-radius: 50%;" in section
    assert "align-items: center;" in section
    assert ".summary-label" in section
    assert ".summary-tab" not in section


def test_strategy_meta_falls_back_to_the_raw_strategy_name():
    meta = ui._strategy_meta("Made Up Strategy")
    assert _COPY_KEYS <= set(meta)
    assert meta["tab"] == "Made Up Strategy"
    # Nothing to say about an unknown strategy and no example query to offer, but
    # everything the form itself renders still has to be there.
    assert all(meta[k] for k in _COPY_KEYS - {"aside_body", "placeholder"})



def test_default_strategy_is_the_first_registered_strategy():
    assert ui._default_strategy() == next(iter(ui.VESPA_STRATEGIES))


# ----- navigation ----------------------------------------------------------
#
# The sidebar navigation is hidden, so the header menu's page links are the only
# way between pages. app.py and that menu must be looking at the same list.

def test_pages_builds_every_page():
    from re_source.nav import pages

    assert len(pages()) == 3


def test_exactly_one_default_page():
    from re_source.nav import pages

    assert sum(1 for p in pages() if p._default) == 1


# ----- going home ----------------------------------------------------------

def test_reset_search_drops_the_whole_search():
    from re_source.search_session import SEARCH_KEYS

    state = {k: "x" for k in SEARCH_KEYS}
    ui._reset_search(state)
    assert state == {}


def test_reset_search_touches_nothing_but_the_search():
    state = {
        "vespa_results": ["r"],
        "export::tips_2024::recA": False,
        "q::Content": "school vouchers",
        "search_strategy_choice": "BM25",
    }
    ui._reset_search(state)
    assert state == {
        "export::tips_2024::recA": False,
        "q::Content": "school vouchers",
        "search_strategy_choice": "BM25",
    }


def test_reset_search_on_a_fresh_session_is_a_no_op():
    state = {}
    ui._reset_search(state)
    assert state == {}


def test_wordmark_clears_the_search_and_switches_to_home(monkeypatch):
    import re_source.nav as nav

    calls = []
    home = object()
    monkeypatch.setattr(ui, "_reset_search", lambda: calls.append("reset"))
    monkeypatch.setattr(nav, "home_page", lambda: home)
    monkeypatch.setattr(ui.st, "switch_page", lambda page: calls.append(page))

    ui._go_home()

    assert calls == ["reset", home]


def test_wordmark_navigates_after_the_button_callback_phase(monkeypatch):
    class _Button:
        def __init__(self):
            self.kwargs = None

        def button(self, *_args, **kwargs):
            self.kwargs = kwargs
            return True

    fake = _Button()
    calls = []
    monkeypatch.setattr(ui, "st", fake)
    monkeypatch.setattr(ui, "_go_home", lambda: calls.append("home"))

    ui._render_home_button()

    assert "on_click" not in fake.kwargs
    assert calls == ["home"]


# ----- filters -------------------------------------------------------------

def test_reset_filters_drops_only_the_filter_widgets():
    state = {
        "filter-datasets": ["tips_2024"],
        "filter-dates": ("2024-01-01", "2024-06-01"),
        "vespa_results": ["r"],
        "q::Content": "school vouchers",
    }
    ui._reset_filters(state)
    assert state == {"vespa_results": ["r"], "q::Content": "school vouchers"}


def test_submitted_date_filter_help_is_plain_and_complete():
    source = inspect.getsource(ui._render_filters)
    assert (
        'help="When the tip was submitted. Rows with no submitted date are "'
        in source
    )
    assert '"excluded when a range is set."' in source
    assert "_created_time" not in source


def test_clearing_the_search_clears_the_filters_too():
    from re_source.search_session import FILTER_KEYS, SEARCH_KEYS

    assert set(FILTER_KEYS) <= set(SEARCH_KEYS)


# ----- the results header ----------------------------------------------------

def test_count_label_names_the_visible_set_only():
    rows = [_FakeResult(f"rec{i}") for i in range(5)]
    assert ui._result_count_label(rows, rows) == "5 responses"
    assert ui._result_count_label(rows[:2], rows) == "2 responses"
    assert ui._result_count_label(rows[:1], rows) == "1 response"


def test_sort_label_survives_a_missing_elapsed():
    assert "-" in ui._result_sort_label(None, "Content")
    assert "2.40s" in ui._result_sort_label(2.4, "Content")
    assert "Content" not in ui._result_sort_label(2.4, "Content")


def test_search_details_is_flush_left_in_the_results_header():
    section = ui._SEARCH_SURFACE_CSS.split("/* --- Results header bar", 1)[1]
    assert "align-self: flex-start;" in section
    assert "margin-left: 0;" in section


# ----- audit hooks ---------------------------------------------------------
#
# _audit_viewed is the only place the UI records what a user was shown, and it
# runs inside button handlers. These exercise it with a dict standing in for
# st.session_state, so no Streamlit runtime is needed.

class _FakeResult:
    def __init__(self, rec, score=0.8742):
        self.source_file = "tips_2024"
        self.doc_id = f"doc-{rec}"
        self.fields = {"_record_id": rec}
        self.score = score
        # The export path reads this, so a stand-in for a result needs it too.
        self.payload = {}


class _FakeSt:
    def __init__(self):
        self.session_state = {}


def _audit_harness(monkeypatch):
    """Swap st for a dict-backed stand-in and capture log_event calls."""
    calls = []
    monkeypatch.setattr(ui, "st", _FakeSt())
    monkeypatch.setattr(ui, "current_user_id", lambda: "dana@example.com")
    monkeypatch.setattr(ui, "session_id", lambda: "sess-1")
    monkeypatch.setattr(ui.audit, "log_event", lambda *a, **kw: calls.append((a, kw)))
    return calls


def test_audit_viewed_records_first_page(monkeypatch):
    calls = _audit_harness(monkeypatch)
    ui._audit_viewed([_FakeResult("recA"), _FakeResult("recB")], "Content",
                     new_search=True)

    (args, kwargs), = calls
    assert args[0] == ui.audit.EVENT_RESULTS_VIEWED
    assert kwargs["record_ids"] == ["tips_2024::recA", "tips_2024::recB"]
    assert kwargs["page"] == 1
    assert kwargs["actor"] == "dana@example.com"
    assert "query" not in kwargs


def test_audit_viewed_load_more_records_only_the_new_slice(monkeypatch):
    calls = _audit_harness(monkeypatch)
    ui._audit_viewed([_FakeResult("recA")], "Content", new_search=True)
    ui._audit_viewed([_FakeResult("recB")], "Content", new_search=False)

    assert [kw["record_ids"] for _, kw in calls] == [
        ["tips_2024::recA"], ["tips_2024::recB"]
    ]
    assert [kw["page"] for _, kw in calls] == [1, 2]


def test_audit_viewed_dedups_within_one_search(monkeypatch):
    calls = _audit_harness(monkeypatch)
    ui._audit_viewed([_FakeResult("recA")], "Content", new_search=True)
    ui._audit_viewed([_FakeResult("recA")], "Content", new_search=False)

    assert len(calls) == 1


def test_audit_viewed_records_same_record_again_for_a_new_search(monkeypatch):
    calls = _audit_harness(monkeypatch)
    ui._audit_viewed([_FakeResult("recA")], "Content", new_search=True)
    ui._audit_viewed([_FakeResult("recA")], "Content", new_search=True)

    assert len(calls) == 2
    assert [kw["page"] for _, kw in calls] == [1, 1]


def test_audit_viewed_partial_overlap_logs_only_unseen(monkeypatch):
    calls = _audit_harness(monkeypatch)
    ui._audit_viewed([_FakeResult("recA")], "Content", new_search=True)
    ui._audit_viewed([_FakeResult("recA"), _FakeResult("recB")], "Content",
                     new_search=False)

    assert calls[1][1]["record_ids"] == ["tips_2024::recB"]


class _FakeStDownload(_FakeSt):
    """st stand-in whose download_button reports a click or not."""

    def __init__(self, clicked):
        super().__init__()
        self.clicked = clicked
        self.label = None
        self.kwargs = None

    def download_button(self, label, **kwargs):
        self.label = label
        self.kwargs = kwargs
        return self.clicked


def _export_harness(monkeypatch, clicked):
    """As _audit_harness, but for the export door."""
    calls = []
    fake = _FakeStDownload(clicked)
    monkeypatch.setattr(ui, "st", fake)
    monkeypatch.setattr(ui, "current_user_id", lambda: "dana@example.com")
    monkeypatch.setattr(ui, "session_id", lambda: "sess-1")
    monkeypatch.setattr(ui.audit, "log_event", lambda *a, **kw: calls.append((a, kw)))
    return fake, calls


def test_export_is_recorded_only_on_a_real_click(monkeypatch):
    _, calls = _export_harness(monkeypatch, clicked=False)
    ui._render_export_button([_FakeResult("recA")], "Content")
    assert calls == []


def test_export_records_the_selected_rows_and_filename(monkeypatch):
    fake, calls = _export_harness(monkeypatch, clicked=True)
    ui._render_export_button(
        [_FakeResult("recA"), _FakeResult("recB")], "Content"
    )

    (args, kwargs), = calls
    assert args[0] == ui.audit.EVENT_EXPORT
    assert kwargs["record_ids"] == ["tips_2024::recA", "tips_2024::recB"]
    assert kwargs["strategy"] == "Content"
    assert "query" not in kwargs
    # The recorded filename is the one the user actually received.
    assert kwargs["filename"] == fake.kwargs["file_name"]
    assert kwargs["filename"].startswith("resource_")


def test_export_button_label_is_just_the_action_and_count(monkeypatch):
    fake, _ = _export_harness(monkeypatch, clicked=False)

    ui._render_export_button(
        [_FakeResult("recA"), _FakeResult("recB")], "Content"
    )

    assert fake.label == "Export 2 selected"


def test_export_with_no_selected_rows_records_nothing(monkeypatch):
    fake, calls = _export_harness(monkeypatch, clicked=True)
    ui._render_export_button([], "Content")
    assert calls == []
    assert fake.label == "Export 0 selected"


# ----- the result card -----------------------------------------------------

def _meta_card(rec="recA", *, submitted=None, dataset="tips_2024"):
    from re_source.result_card import ResultCard

    return ResultCard(
        index=3, ident="x", dataset=dataset, submitted=submitted, url=None,
        matches=(), fallback="", matched_fields="—", highlights=(),
        why="—", evidence_label="Top matching chunks", included=True, fields={},
    )


def test_card_meta_line_carries_number_base_and_date():
    from datetime import date

    out = ui._card_meta_html(_meta_card(submitted=date(2024, 11, 13)))
    assert ">03<" in out
    assert "tips_2024" in out
    assert "submitted 2024-11-13" in out


def test_card_meta_line_never_shows_a_score():
    from datetime import date

    out = ui._card_meta_html(_meta_card(submitted=date(2024, 11, 13)))
    assert "score" not in out and "0.874" not in out


def test_card_meta_line_survives_a_missing_date():
    out = ui._card_meta_html(_meta_card())
    assert "submitted" not in out


def test_card_meta_line_escapes_the_base_name():
    out = ui._card_meta_html(_meta_card(dataset="tips & <b>leaks</b>"))
    assert "<b>" not in out and "&amp;" in out


# ----- nothing to show -----------------------------------------------------

class _FakeStEmpty(_FakeSt):
    def __init__(self):
        super().__init__()
        self.written = []

    def markdown(self, body, **kw):
        self.written.append(body)


def test_empty_state_is_silent_when_there_are_cards(monkeypatch):
    fake = _FakeStEmpty()
    monkeypatch.setattr(ui, "st", fake)
    ui._render_empty_state(["r"], ["r"])
    assert fake.written == []


def test_empty_state_blames_the_filters_when_results_exist(monkeypatch):
    fake = _FakeStEmpty()
    monkeypatch.setattr(ui, "st", fake)
    ui._render_empty_state(["a", "b", "c"], [])
    (msg,) = fake.written
    assert "filters" in msg and "all 3" in msg


def test_a_search_that_found_nothing_shows_no_message(monkeypatch):
    fake = _FakeStEmpty()
    monkeypatch.setattr(ui, "st", fake)
    ui._render_empty_state([], [])
    assert fake.written == []


def test_no_strategy_carries_empty_state_advice():
    assert not any("empty" in meta for meta in ui._STRATEGY_UI.values())
    assert "empty" not in ui._STRATEGY_UI_FALLBACK


# ----- running a search ------------------------------------------------------
#
# _execute_search used to write seven session-state keys by hand, which meant
# there was no point at which "one search" existed as a thing to assert on.


class _FakeStSearch(_FakeSt):
    """st stand-in for the search path: a no-op progress bar, and stop() that
    raises so an error path can be told from a success."""

    class _Progress:
        def progress(self, *a, **kw):
            pass

        def empty(self):
            pass

    def __init__(self):
        super().__init__()
        self.errors = []

    def progress(self, *a, **kw):
        return self._Progress()

    def error(self, msg):
        self.errors.append(msg)

    def stop(self):
        raise RuntimeError("st.stop")


class _FakeStrategy:
    """A paginating strategy whose pool is deeper than one page."""

    PAGE_SIZE = 2
    paginates = True

    def __init__(self, pool=None, page=None):
        self._pool = pool if pool is not None else [f"c{i}" for i in range(5)]
        self._page = page

    def retrieve_pool(self, query, top_k):
        return self._pool, {"num_candidates": len(self._pool)}

    def validate_page(self, query, pool_slice):
        if self._page is not None:
            return list(self._page)
        return [_FakeResult(f"rec{c}") for c in pool_slice]


class _FakeOneShot:
    """A strategy that returns everything at once — no pool, no Load more."""

    paginates = False

    def execute(self, query, top_k):
        class _R:
            results = [_FakeResult("recA")]
            metadata = {"num_parents": 1}
        return _R()


def _run_for(strategy, name="Content"):
    """A real SearchRun around a stand-in strategy — the run module is what the
    UI talks to now, so the tests go through it too."""
    from re_source.search_runner import SearchRun

    return SearchRun(name, strategy)


def _search_harness(monkeypatch, strategy, name="Content"):
    calls = []
    fake = _FakeStSearch()
    monkeypatch.setattr(ui, "st", fake)
    monkeypatch.setattr(ui, "current_user_id", lambda: "dana@example.com")
    monkeypatch.setattr(ui, "session_id", lambda: "sess-1")
    monkeypatch.setattr(ui, "start_run",
                        lambda n, **kw: _run_for(strategy, n))
    monkeypatch.setattr(ui, "propagate_attributes", _null_context)
    monkeypatch.setattr(ui.audit, "log_event", lambda *a, **kw: calls.append((a, kw)))
    return fake, calls


class _null_context:
    def __init__(self, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_execute_search_writes_one_coherent_search(monkeypatch):
    fake, _ = _search_harness(monkeypatch, _FakeStrategy())
    ui._execute_search("Content", "mold")

    active = ui._session().active()
    assert active.query == "mold"
    assert active.strategy == "Content"
    assert active.elapsed is not None
    assert len(active.results) == 2          # one page of PAGE_SIZE
    assert active.pool == ["c0", "c1", "c2", "c3", "c4"]
    assert active.cursor == 2


def test_execute_search_audits_the_first_page(monkeypatch):
    _, calls = _search_harness(monkeypatch, _FakeStrategy())
    ui._execute_search("Content", "mold")

    (args, kwargs), = calls
    assert args[0] == ui.audit.EVENT_RESULTS_VIEWED
    assert kwargs["page"] == 1
    assert "query" not in kwargs


def test_execute_search_clears_the_previous_filters(monkeypatch):
    fake, _ = _search_harness(monkeypatch, _FakeStrategy())
    fake.session_state["filter-datasets"] = ["tips_2024"]

    ui._execute_search("Content", "mold")

    assert "filter-datasets" not in fake.session_state


def test_a_one_shot_strategy_stores_no_pool(monkeypatch):
    fake, _ = _search_harness(monkeypatch, _FakeOneShot())
    ui._execute_search("BM25", "Experian")

    session = ui._session()
    assert session.active().pool is None
    assert session.page_plan(50) is None


# ----- going deeper ----------------------------------------------------------
#
# The whole Load more handler used to sit inside `if clicked:`, so none of it
# ran under a test: not the cursor advance, not the slice bounds, and not the
# rule that only the newly-revealed records are recorded as disclosed.


def _loaded(monkeypatch, strategy, *, pool, cursor, results):
    """A session mid-search, with Load more applied once."""
    fake, calls = _search_harness(monkeypatch, strategy)
    session = ui._session()
    session.start(results=list(results), metadata={"validated_count": len(results)},
                  elapsed=1.0, query="mold", strategy="Content",
                  pool=list(pool), cursor=cursor)
    active = session.active()
    run = _run_for(strategy)
    plan = session.page_plan(run.page_size)
    ui._apply_load_more(session, run, active, plan)
    return session, calls, plan


def test_load_more_validates_the_next_slice_not_the_same_one_again(monkeypatch):
    strategy = _FakeStrategy()
    session, _, plan = _loaded(
        monkeypatch, strategy,
        pool=["c0", "c1", "c2", "c3", "c4"], cursor=2, results=[_FakeResult("recA")],
    )
    assert plan.pool_slice == ["c2", "c3"]
    assert session.active().cursor == 4


def test_load_more_appends_rather_than_replacing(monkeypatch):
    session, _, _ = _loaded(
        monkeypatch, _FakeStrategy(page=[_FakeResult("recB")]),
        pool=["c0", "c1", "c2", "c3"], cursor=2, results=[_FakeResult("recA")],
    )
    ids = [r.fields["_record_id"] for r in session.active().results]
    assert ids == ["recA", "recB"]


def test_load_more_tracks_the_running_total(monkeypatch):
    session, _, _ = _loaded(
        monkeypatch, _FakeStrategy(page=[_FakeResult("recB"), _FakeResult("recC")]),
        pool=["c0", "c1", "c2", "c3"], cursor=2, results=[_FakeResult("recA")],
    )
    assert session.active().metadata["validated_count"] == 3


def test_load_more_audits_only_the_new_slice(monkeypatch):
    _, calls, _ = _loaded(
        monkeypatch, _FakeStrategy(page=[_FakeResult("recB")]),
        pool=["c0", "c1", "c2", "c3"], cursor=2, results=[_FakeResult("recA")],
    )
    (args, kwargs), = calls
    assert args[0] == ui.audit.EVENT_RESULTS_VIEWED
    assert kwargs["record_ids"] == ["tips_2024::recB"]
    assert kwargs["page"] == 2


def test_load_more_does_not_re_audit_a_record_already_recorded(monkeypatch):
    fake, calls = _search_harness(monkeypatch, _FakeStrategy())
    session = ui._session()
    session.start(results=[], metadata={}, elapsed=1.0, query="mold",
                  strategy="Content", pool=["c0", "c1"], cursor=0)
    ui._audit_viewed([_FakeResult("recA")], "Content", new_search=True)

    run = _run_for(_FakeStrategy(page=[_FakeResult("recA")]))
    ui._apply_load_more(session, run, session.active(),
                        session.page_plan(run.page_size))

    assert len(calls) == 1            # the first page only
    assert session.active().cursor == 2   # ...but the pool still advanced


def test_the_last_page_is_short_rather_than_overrunning(monkeypatch):
    session, _, plan = _loaded(
        monkeypatch, _FakeStrategy(),
        pool=["c0", "c1", "c2"], cursor=2, results=[],
    )
    assert plan.step == 1
    assert session.active().cursor == 3


def test_there_is_nothing_more_to_load_once_the_pool_is_spent(monkeypatch):
    fake, _ = _search_harness(monkeypatch, _FakeStrategy())
    session = ui._session()
    session.start(results=[], metadata={}, elapsed=1.0, query="mold",
                  strategy="Content", pool=["c0", "c1"], cursor=2)
    assert session.page_plan(2) is None


# ----- the shared results page -----------------------------------------------
#
# Both views emitted the same ~25 lines of scaffold and differed only in the
# per-result loop. The module docstring said so from the start; now there is a
# seam that makes it true rather than a description of two copies.


class _FakeStPage(_FakeSt):
    """Records which containers and columns were opened, in order."""

    class _Ctx:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def __init__(self):
        super().__init__()
        self.keys = []
        self.markdown_calls = []

    def container(self, **kw):
        if "key" in kw:
            self.keys.append(kw["key"])
        return self._Ctx()

    def columns(self, spec, **kw):
        return [self._Ctx() for _ in spec]

    def markdown(self, body, **kw):
        self.markdown_calls.append(body)


def _page_harness(monkeypatch):
    fake = _FakeStPage()
    monkeypatch.setattr(ui, "st", fake)
    monkeypatch.setattr(ui, "_render_search_controls", lambda: None)
    monkeypatch.setattr(ui, "_render_filters", lambda results: list(results))
    monkeypatch.setattr(ui, "_render_results_header", lambda *a, **kw: None)
    monkeypatch.setattr(ui, "_render_empty_state", lambda *a, **kw: None)
    monkeypatch.setattr(ui, "_render_load_more", lambda: None)
    return fake


def test_the_results_page_is_silent_before_a_search(monkeypatch):
    fake = _page_harness(monkeypatch)
    drawn = []
    ui._render_results_page(result_css="", render_result=lambda *a: drawn.append(a))

    assert drawn == []
    assert fake.keys == []


def test_the_results_page_draws_one_row_per_filtered_result(monkeypatch):
    fake = _page_harness(monkeypatch)
    rows = [_FakeResult("recA"), _FakeResult("recB"), _FakeResult("recC")]
    ui._session().start(results=rows, metadata={}, elapsed=1.0, query="mold",
                        strategy="Content")

    drawn = []
    ui._render_results_page(result_css="", render_result=lambda i, r, sel: drawn.append((i, r)))

    assert [i for i, _ in drawn] == [1, 2, 3]
    assert [r.fields["_record_id"] for _, r in drawn] == ["recA", "recB", "recC"]


def test_every_row_is_drawn_against_the_same_selection(monkeypatch):
    _page_harness(monkeypatch)
    rows = [_FakeResult("recA"), _FakeResult("recB")]
    ui._session().start(results=rows, metadata={}, elapsed=1.0, query="mold",
                        strategy="Content")

    seen = []
    ui._render_results_page(result_css="", render_result=lambda i, r, sel: seen.append(sel))

    assert seen[0] is seen[1]


def test_the_view_css_is_emitted_before_the_visible_results_stack(monkeypatch):
    fake = _page_harness(monkeypatch)
    ui._session().start(results=[_FakeResult("recA")], metadata={}, elapsed=1.0,
                        query="mold", strategy="Content")

    order = []
    monkeypatch.setattr(ui, "_render_results_header",
                        lambda *a, **kw: order.append("header"))
    monkeypatch.setattr(ui, "_render_empty_state",
                        lambda *a, **kw: order.append("empty"))
    fake.markdown = lambda body, **kw: order.append("css") if body == "SENTINEL" else None

    ui._render_results_page(result_css="SENTINEL",
                            render_result=lambda *a: order.append("row"))

    assert order == ["css", "header", "empty", "row"]
    assert "results-list" in fake.keys


def test_result_list_and_cards_use_the_requested_spacing():
    shared = ui._SEARCH_SURFACE_CSS.split("/* --- Results header bar", 1)[1]
    assert "margin-bottom: 0;" in shared
    assert "gap: 16px;" in shared
    assert "padding: 20px 24px;" in ui._RESULT_CARD_CSS


def test_both_pages_render_through_the_one_scaffold(monkeypatch):
    calls = []
    monkeypatch.setattr(ui, "_render_results_page",
                        lambda **kw: calls.append(kw["render_result"]))

    ui.render_vespa_search()
    ui.render_vespa_table()

    assert calls == [ui._render_result_card, ui._render_result_columns]


def test_a_non_paginating_search_draws_no_load_more_and_builds_no_strategy(monkeypatch):
    fake = _FakeStPage()
    monkeypatch.setattr(ui, "st", fake)

    def _no_run(name, **kw):
        raise AssertionError("a non-paginating search built a strategy")

    monkeypatch.setattr(ui, "start_run", _no_run)
    ui._session().start(results=[_FakeResult("recA")], metadata={}, elapsed=1.0,
                        query="Experian", strategy="BM25")   # no pool

    ui._render_load_more()                      # must not raise

    assert fake.keys == []                      # and draws nothing


def test_the_load_more_guard_holds_for_every_registered_strategy(monkeypatch):
    from re_source.vespa_strategies import VESPA_STRATEGIES

    for name, cls in VESPA_STRATEGIES.items():
        fake = _FakeStPage()
        monkeypatch.setattr(ui, "st", fake)
        monkeypatch.setattr(ui, "start_run",
                            lambda n, c=cls, **kw: _run_for(c(object()), n))
        ui._session().start(results=[_FakeResult("recA")], metadata={},
                            elapsed=1.0, query="q", strategy=name)
        ui._render_load_more()                  # must not raise for any strategy

# ----- the login gate --------------------------------------------------------
#
# _enforce_login is the app's authentication gate and had no tests at all,
# because REQUIRE_LOGIN was bound at import time and could not be monkeypatched
# the way re_source/audit.py's config reads can. It reads config at call time now,
# matching that convention, which is what makes these possible.


class _Stopped(Exception):
    """Raised by the fake st.stop() so a halt can be told from a pass."""


class _FakeStLogin(_FakeSt):
    def __init__(self, *, logged_in=False, has_auth=True):
        super().__init__()
        self.secrets = {"auth": {}} if has_auth else {}
        self.user = type("U", (), {"is_logged_in": logged_in})()
        self.messages = []
        self.buttons = []

    def button(self, label, **kw):
        self.buttons.append((label, kw))
        return False

    def login(self):
        pass

    def error(self, msg):
        self.messages.append(("error", msg))

    def info(self, msg):
        self.messages.append(("info", msg))

    def stop(self):
        raise _Stopped


def _login_harness(monkeypatch, *, require, logged_in=False, has_auth=True):
    fake = _FakeStLogin(logged_in=logged_in, has_auth=has_auth)
    monkeypatch.setattr(ui, "st", fake)
    monkeypatch.setattr(ui.config, "REQUIRE_LOGIN", require)
    return fake


def test_an_anonymous_visitor_is_stopped_when_login_is_required(monkeypatch):
    fake = _login_harness(monkeypatch, require=True, logged_in=False)
    with pytest.raises(_Stopped):
        ui._enforce_login()
    assert fake.messages == [("info", "Please log in to search.")]


def test_a_signed_in_user_passes(monkeypatch):
    _login_harness(monkeypatch, require=True, logged_in=True)
    ui._enforce_login()          # does not raise


def test_login_not_required_lets_anyone_through(monkeypatch):
    _login_harness(monkeypatch, require=False, logged_in=False, has_auth=False)
    ui._enforce_login()


def test_requiring_login_without_an_auth_provider_fails_closed(monkeypatch):
    fake = _login_harness(monkeypatch, require=True, logged_in=False, has_auth=False)
    with pytest.raises(_Stopped):
        ui._enforce_login()
    (kind, msg), = fake.messages
    assert kind == "error" and "[auth]" in msg


def test_an_unreadable_secrets_file_fails_closed(monkeypatch):
    fake = _login_harness(monkeypatch, require=True, logged_in=False)

    class _Exploding:
        def __contains__(self, key):
            raise RuntimeError("no secrets file")

    fake.secrets = _Exploding()
    with pytest.raises(_Stopped):
        ui._enforce_login()


def test_a_user_object_without_the_flag_is_not_logged_in(monkeypatch):
    fake = _login_harness(monkeypatch, require=True)
    fake.user = object()
    with pytest.raises(_Stopped):
        ui._enforce_login()


def test_a_locked_out_visitor_is_given_a_way_in(monkeypatch):
    fake = _login_harness(monkeypatch, require=True, logged_in=False)

    with pytest.raises(_Stopped):
        ui._enforce_login()

    assert any("log in" in label.lower() for label, _ in fake.buttons), fake.buttons


def test_the_way_in_actually_signs_you_in(monkeypatch):
    fake = _login_harness(monkeypatch, require=True, logged_in=False)

    with pytest.raises(_Stopped):
        ui._enforce_login()

    (_, kwargs), = fake.buttons
    assert kwargs["on_click"] == fake.login


def test_a_misconfigured_gate_offers_no_login_it_cannot_honour(monkeypatch):
    fake = _login_harness(monkeypatch, require=True, has_auth=False)

    with pytest.raises(_Stopped):
        ui._enforce_login()

    assert fake.buttons == []


# ----- the filter panel's own wiring -------------------------------------------
#
# result_filter.py's docstring says the predicate "deserves to be callable and
# tested, rather than reachable only by executing four Streamlit widget calls —
# which is how it was, and why the one test that touched the results page
# replaced it with `lambda results: list(results)`." The arithmetic moved out
# and the wiring stayed, untested, in a render function. This covers the wiring:
# the ordering reconcile depends on, the keys the widgets carry, and what the
# panel says when a range is half-picked.

class _FakeStFilters(_FakeSt):
    """Enough Streamlit for _render_filters, recording the call order."""

    def __init__(self, selected=(), dates=()):
        super().__init__()
        self.calls = []
        self._selected = list(selected)
        self._dates = dates
        self.captions = []
        self.widget_keys = {}

    def markdown(self, *a, **kw):
        self.calls.append("markdown")

    def multiselect(self, label, options, default=None, key=None, help=None):
        self.calls.append("multiselect")
        self.widget_keys["multiselect"] = key
        return self._selected or list(options)

    def date_input(self, label, value=None, key=None, help=None):
        self.calls.append("date_input")
        self.widget_keys["date_input"] = key
        return self._dates

    def button(self, *a, **kw):
        self.calls.append("button")
        return False

    def caption(self, text):
        self.calls.append("caption")
        self.captions.append(text)

    def checkbox(self, label, key=None, **kw):
        self.calls.append("checkbox")
        self.widget_keys.setdefault("checkbox", []).append((label, key))
        return bool(self.session_state.get(key))


def _filters(monkeypatch, fake, rows):
    monkeypatch.setattr(ui, "st", fake)
    return ui._render_filters(rows)


def test_the_selection_is_reconciled_before_the_widget_draws(monkeypatch):
    """The ordering constraint that used to live only in a comment: the widget
    reads its own key as state, so a write after it renders is a fight."""
    from re_source.result_filter import DATASETS_KEY, OFFERED_KEY

    fake = _FakeStFilters()
    fake.session_state[DATASETS_KEY] = ["tips_2024"]
    fake.session_state[OFFERED_KEY] = ["tips_2024"]

    _filters(monkeypatch, fake, [_FakeResult("recA"), _FakeResult("recB")])

    # reconcile widened the stored selection, and it did so before the widget.
    assert fake.calls.index("multiselect") > 0
    assert fake.session_state[OFFERED_KEY] == ["tips_2024"]


def test_the_widgets_carry_the_keys_result_filter_reconciles(monkeypatch):
    """A mismatch here silently discards the reconciled selection."""
    from re_source.result_filter import DATASETS_KEY, DATES_KEY

    fake = _FakeStFilters()
    _filters(monkeypatch, fake, [_FakeResult("recA")])

    assert fake.widget_keys["multiselect"] == DATASETS_KEY
    assert fake.widget_keys["date_input"] == DATES_KEY


def test_filter_panel_has_the_requested_spacing_and_alignment():
    section = ui._SEARCH_SURFACE_CSS.split("/* --- Filter panel", 1)[1]
    source = inspect.getsource(ui._render_filters)
    page = inspect.getsource(ui._render_results_page)

    assert "padding: 12px 20px 8px;" in section
    assert "margin-bottom: 12px;" in section
    assert "align-items: flex-start;" in section
    assert "use_container_width=True" not in source
    assert 'vertical_alignment="top"' in page


def test_a_half_picked_range_is_called_out(monkeypatch):
    from datetime import date

    fake = _FakeStFilters(dates=(date(2026, 1, 1),))
    _filters(monkeypatch, fake, [_FakeResult("recA")])

    assert any("end date" in c for c in fake.captions)


def test_a_complete_range_is_not_called_out(monkeypatch):
    from datetime import date

    fake = _FakeStFilters(dates=(date(2020, 1, 1), date(2030, 1, 1)))
    _filters(monkeypatch, fake, [_FakeResult("recA")])

    assert not any("end date" in c for c in fake.captions)


def test_the_panel_returns_the_narrowed_rows(monkeypatch):
    """What this returns is what the export builds its workbook from, and what
    the audit log records as disclosed."""
    fake = _FakeStFilters(selected=["tips_2024"])
    rows = [_FakeResult("recA"), _FakeResult("recB")]

    assert _filters(monkeypatch, fake, rows) == rows


def test_deselecting_every_base_shows_nothing(monkeypatch):
    """An empty tuple means "the reporter removed them all", distinct from
    None meaning "every dataset"."""
    fake = _FakeStFilters()
    fake._selected = []
    monkeypatch.setattr(ui, "st", fake)

    # multiselect returns options when _selected is empty, so force the case.
    fake.multiselect = lambda *a, **kw: []
    assert ui._render_filters([_FakeResult("recA")]) == []


def test_no_filter_policy_draws_no_group_checkbox(monkeypatch):
    fake = _FakeStFilters()
    _filters(monkeypatch, fake, [_FakeResult("recA")])

    assert "checkbox" not in fake.calls


def test_a_ticked_group_checkbox_hides_its_datasets_everywhere(monkeypatch):
    """Hidden rows leave the view and the export, and the dataset menu stops
    offering them, so the checkbox and the menu can't disagree."""
    from re_source import result_filter
    from re_source.result_filter import GroupExclusion, OFFERED_KEY

    exclusion = GroupExclusion("Exclude active callouts", "wsp_callouts")
    monkeypatch.setattr(result_filter, "group_exclusions", lambda: [exclusion])
    monkeypatch.setattr(result_filter, "dataset_group",
                        {"app_callout": "wsp_callouts"}.get)
    callout, kept = _FakeResult("recA"), _FakeResult("recB")
    callout.source_file = "app_callout"

    fake = _FakeStFilters()
    shown = _filters(monkeypatch, fake, [callout, kept])
    assert shown == [callout, kept]
    assert fake.widget_keys["checkbox"] == [("Exclude active callouts", exclusion.key)]
    assert fake.calls.index("checkbox") < fake.calls.index("multiselect")

    fake = _FakeStFilters()
    fake.session_state[exclusion.key] = True
    assert _filters(monkeypatch, fake, [callout, kept]) == [kept]
    assert "app_callout" not in fake.session_state[OFFERED_KEY]
