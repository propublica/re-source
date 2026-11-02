"""The admin upload page imports cleanly and its validation rules hold."""

from contextlib import nullcontext
import inspect
from types import SimpleNamespace

import pandas as pd
import pytest

import re_source.admin_ui as admin


def test_render_is_callable():
    assert callable(admin.render_admin_upload)


def test_upload_page_checks_admin_permission():
    source = inspect.getsource(admin.render_admin_upload)
    assert "is_admin(st.user)" in source


def test_failed_index_feed_does_not_report_success():
    source = inspect.getsource(admin._render_feed_step)
    assert 'status.update(label="Indexing incomplete", state="error")' in source
    assert "if failed:" in source


def test_upload_destination_is_derived_from_the_filename_not_editable():
    source = inspect.getsource(admin._render_import_step)

    assert ".text_input(" not in source
    assert "table = suggest_table_name(uploaded.name)" in source


def test_admin_css_does_not_restyle_buttons_or_checkboxes(monkeypatch):
    rendered = []
    monkeypatch.setattr(
        admin.st,
        "markdown",
        lambda body, **_kwargs: rendered.append(body),
    )

    admin._inject_css()

    css = rendered[0].lower()
    assert "button" not in css
    assert "checkbox" not in css


class _EnrichmentStepSt:
    def __init__(self):
        self.buttons = []

    def container(self, **_kwargs):
        return nullcontext()

    def markdown(self, *_args, **_kwargs):
        pass

    def caption(self, *_args, **_kwargs):
        pass

    def button(self, label, **_kwargs):
        self.buttons.append(label)
        return False


class _ClickedEnrichmentStepSt(_EnrichmentStepSt):
    def __init__(self):
        super().__init__()
        self.status_calls = 0
        self.spinner_calls = 0

    def button(self, label, **_kwargs):
        self.buttons.append(label)
        return True

    def status(self, *_args, **_kwargs):
        self.status_calls += 1

        class _Status:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                pass

            def update(self, **_kwargs):
                pass

        return _Status()

    def spinner(self, *_args, **_kwargs):
        self.spinner_calls += 1
        return nullcontext()

    def success(self, *_args, **_kwargs):
        pass

    def warning(self, *_args, **_kwargs):
        pass

    def error(self, *_args, **_kwargs):
        pass


def test_author_enrichment_is_available_after_upload(monkeypatch):
    fake = _EnrichmentStepSt()
    monkeypatch.setattr(admin, "st", fake)

    admin._render_enrichment_step("tips")

    assert fake.buttons == ["Run author enrichment"]


def test_author_enrichment_uses_a_spinner_not_an_empty_status_panel(monkeypatch):
    fake = _ClickedEnrichmentStepSt()
    monkeypatch.setattr(admin, "st", fake)
    monkeypatch.setattr(
        admin,
        "_run_author_enrichment",
        lambda _table: (
            {
                "failed": 0,
                "scanned": 1,
                "enriched": 1,
                "skipped_unchanged": 0,
                "nonempty_profiles": 1,
            },
            {"fed_chunks": 1},
        ),
    )

    admin._render_enrichment_step("tips")

    assert fake.spinner_calls == 1
    assert fake.status_calls == 0


def test_author_enrichment_is_scoped_to_the_uploaded_table(monkeypatch):
    import re_source.db_loader as db_loader
    import re_source.enrich as enrich
    import re_source.vespa_client as vespa_client

    wanted = SimpleNamespace(source_file="tips")
    other = SimpleNamespace(source_file="other")
    client = object()
    calls = []
    monkeypatch.setattr(db_loader, "get_all_documents", lambda: [wanted, other])
    monkeypatch.setattr(
        enrich,
        "enrich_rows",
        lambda docs: calls.append(("compute", docs)) or {"enriched": 1},
    )
    monkeypatch.setattr(
        enrich,
        "feed_enrichment",
        lambda got_client, docs: calls.append(("feed", got_client, docs))
        or {"fed_chunks": 1},
    )
    monkeypatch.setattr(vespa_client, "get_vespa_client", lambda: client)

    assert admin._run_author_enrichment("tips") == (
        {"enriched": 1},
        {"fed_chunks": 1},
    )
    assert calls == [("compute", [wanted]), ("feed", client, [wanted])]


# ----- the memoised plan -----------------------------------------------------
#
# The plan is memoised against the file and the destination, because building
# one reads every table in the database and Streamlit reruns on every
# keystroke. Neither of those changes when the import runs — so the import has
# to say so itself, or the page keeps answering with a plan that was true
# before the table existed.


class _FakeSt:
    def __init__(self):
        self.session_state = {}


def test_a_successful_import_invalidates_the_plan(monkeypatch):
    fake = _FakeSt()
    monkeypatch.setattr(admin, "st", fake)
    fake.session_state[admin._DELTA] = {"key": ("tips", "id", "f"), "value": "stale"}

    admin._forget_plan()

    assert admin._DELTA not in fake.session_state


def test_forgetting_a_plan_that_was_never_built_is_harmless(monkeypatch):
    fake = _FakeSt()
    monkeypatch.setattr(admin, "st", fake)
    admin._forget_plan()          # must not raise
    assert fake.session_state == {}


def test_the_memo_is_rebuilt_after_it_is_forgotten(monkeypatch):
    fake = _FakeSt()
    monkeypatch.setattr(admin, "st", fake)
    built = []
    monkeypatch.setattr(admin, "build_import_plan",
                        lambda *a, **kw: built.append(1) or f"plan-{len(built)}")

    first = admin._cached_plan(None, "id", "tips", "f")
    assert admin._cached_plan(None, "id", "tips", "f") is first   # memoised
    assert len(built) == 1

    admin._forget_plan()

    assert admin._cached_plan(None, "id", "tips", "f") != first   # rebuilt
    assert len(built) == 2


# ----- the uploaded file does not outlive the parse --------------------------
#
# The temp file holds reader tips in plaintext. `tmp` used to be bound as the
# last statement of the with-block, so a failure during the write left it
# unbound while the file already existed — and the finally that deletes it was
# never reached. The user is told "Could not read that file", i.e. a failure
# with no residue, and nothing here ever sweeps that directory.

class _Uploaded:
    """Enough of a Streamlit UploadedFile for _parse_upload."""

    def __init__(self, name, data=b"a,b\n1,2\n", raises=None):
        self.name = name
        self._data = data
        self._raises = raises

    def getbuffer(self):
        if self._raises:
            raise self._raises
        return self._data


def _temp_files(d):
    from pathlib import Path
    return set(Path(d).iterdir())


def test_a_successful_parse_leaves_no_file_behind(tmp_path, monkeypatch):
    import tempfile as _t
    monkeypatch.setattr(_t, "tempdir", str(tmp_path))
    monkeypatch.setenv("TMPDIR", str(tmp_path))

    before = _temp_files(tmp_path)
    admin._parse_upload(_Uploaded("tips.csv"), None)
    assert _temp_files(tmp_path) == before


def test_a_write_failure_leaves_no_file_behind(tmp_path, monkeypatch):
    """The regression: the exception arrives before the old binding existed."""
    import tempfile as _t
    monkeypatch.setattr(_t, "tempdir", str(tmp_path))
    monkeypatch.setenv("TMPDIR", str(tmp_path))

    before = _temp_files(tmp_path)
    with pytest.raises(MemoryError):
        admin._parse_upload(
            _Uploaded("tips.csv", raises=MemoryError("no room")), None
        )
    assert _temp_files(tmp_path) == before


def test_a_parse_failure_leaves_no_file_behind(tmp_path, monkeypatch):
    import tempfile as _t
    monkeypatch.setattr(_t, "tempdir", str(tmp_path))
    monkeypatch.setenv("TMPDIR", str(tmp_path))

    before = _temp_files(tmp_path)
    with pytest.raises(Exception):
        admin._parse_upload(_Uploaded("tips.xlsx", data=b"not a workbook"), None)
    assert _temp_files(tmp_path) == before
