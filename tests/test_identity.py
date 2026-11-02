"""User identity resolution + Langfuse attribution wiring (no Streamlit runtime)."""

import re_source.search_ui as ui


def test_user_id_from_email():
    assert ui._user_id_from({"email": "a@x.com"}) == "a@x.com"


def test_user_id_from_empty_is_anonymous():
    assert ui._user_id_from({}) == "anonymous"


def test_user_id_from_none_email_falls_back_not_anonymous():
    assert ui._user_id_from({"email": None, "sub": "00u1abc"}) == "00u1abc"


def test_preferred_username_is_preferred_over_sub():
    user = {"preferred_username": "jane@example.com", "sub": "00u1abc"}
    assert ui._user_id_from(user) == "jane@example.com"


def test_email_still_wins_when_present():
    user = {"email": "a@x.com", "preferred_username": "b@x.com", "sub": "00u1abc"}
    assert ui._user_id_from(user) == "a@x.com"


def test_blank_claims_are_skipped():
    assert ui._user_id_from({"email": "", "sub": "00u1abc"}) == "00u1abc"


def test_no_identifying_claim_at_all_is_anonymous():
    assert ui._user_id_from({"email": None, "sub": None}) == "anonymous"


class _RecordingCM:
    """Stand-in for langfuse.propagate_attributes: records kwargs, is a CM."""

    def __init__(self):
        self.kwargs = None

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeOneShotStrategy:
    """A strategy that answers in one pass, like BM25."""

    paginates = False

    def __init__(self):
        self.called_with = None

    def execute(self, query, top_k):
        self.called_with = (query, top_k)

        class _R:
            results = ["RESULT"]
            metadata = {}
        return _R()


def test_the_ui_stamps_a_trace_with_who_is_searching(monkeypatch):
    rec = _RecordingCM()
    monkeypatch.setattr(ui, "propagate_attributes", rec)
    monkeypatch.setattr(ui, "current_user_id", lambda: "jane@example.com")
    monkeypatch.setattr(ui, "session_id", lambda: "sess-123")

    with ui._attributed(strategy="BM25", top_k=5):
        pass

    assert rec.kwargs["user_id"] == "jane@example.com"
    assert rec.kwargs["session_id"] == "sess-123"
    assert rec.kwargs["metadata"]["strategy"] == "BM25"
    assert rec.kwargs["metadata"]["top_k"] == 5
    assert rec.kwargs["metadata"]["surface"] == "streamlit"


def test_a_run_executes_inside_the_attribution_it_was_given(monkeypatch):
    from re_source.search_runner import SearchRun

    rec = _RecordingCM()
    strat = _FakeOneShotStrategy()
    run = SearchRun("BM25", strat, attribution=rec)

    page = run.first_page("teachers", top_k=5)

    assert page.results == ["RESULT"]
    assert strat.called_with == ("teachers", 5)
    assert rec.kwargs == {"strategy": "BM25", "top_k": 5}


def test_a_run_without_attribution_still_runs():
    from re_source.search_runner import SearchRun

    run = SearchRun("BM25", _FakeOneShotStrategy())
    assert run.first_page("teachers").results == ["RESULT"]
