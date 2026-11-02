"""Which columns get chunked, and whether the question rides along."""

import pytest

from re_source import chunker
from re_source.chunker import chunk_document, chunk_mode
from re_source.db_loader import Document


@pytest.fixture(autouse=True)
def _no_inventory(monkeypatch):
    """Default: no field_inventory table, so tests state their own classes."""
    monkeypatch.setattr(chunker, "_field_classes_cache", {})
    monkeypatch.setattr(chunker, "_reviewed_field_classes_cache", set())


def _classify(monkeypatch, mapping):
    monkeypatch.setattr(chunker, "_field_classes_cache", mapping)


def _doc(fields):
    return Document(id="tips__rec1", source_file="tips", fields=fields,
                    content_hash="h", text="")


# ----- the three modes -------------------------------------------------------


def test_public_default_does_not_exclude_classified_fields(monkeypatch):
    _classify(monkeypatch, {("tips", "Your name"): "contact_or_id",
                            ("tips", "Submitted"): "metadata"})
    chunks, _ = chunk_document(_doc({"Your name": "Jane Doe", "Submitted": "2026-01-01"}))
    assert [chunk.column_name for chunk in chunks] == ["Your name", "Submitted"]


def test_free_text_chunks_the_answer_alone(monkeypatch):
    _classify(monkeypatch, {("tips", "How has your insurance affected you?"): "content_free_text"})
    chunks, _ = chunk_document(
        _doc({"How has your insurance affected you?": "I was denied coverage twice."})
    )
    assert [c.text for c in chunks] == ["I was denied coverage twice."]
    # The question is not lost — it is on the chunk as column_name.
    assert chunks[0].column_name == "How has your insurance affected you?"


def test_closed_ended_keeps_the_question(monkeypatch):
    _classify(monkeypatch, {("tips", "Are you a teacher?"): "content_closed_ended"})
    chunks, _ = chunk_document(_doc({"Are you a teacher?": "No"}))
    assert [c.text for c in chunks] == ["Are you a teacher? No"]


def test_needs_review_keeps_the_question(monkeypatch):
    _classify(monkeypatch, {("tips", "Which agency?"): "needs_review"})
    chunks, _ = chunk_document(_doc({"Which agency?": "FEMA"}))
    assert [c.text for c in chunks] == ["Which agency? FEMA"]


# ----- fallback when nothing is classified -----------------------------------


def test_public_default_keeps_unclassified_contact_fields():
    assert chunk_mode("tips", "email") == "with_question"
    assert chunk_mode("tips", "What happened?") == "with_question"


def test_an_unclassified_column_still_chunks_with_its_question():
    chunks, _ = chunk_document(_doc({"What happened?": "A long story."}))
    assert [c.text for c in chunks] == ["What happened? A long story."]


def test_classification_can_choose_answer_only(monkeypatch):
    _classify(monkeypatch, {("tips", "Did you email the district?"): "content_closed_ended"})
    assert chunk_mode("tips", "Did you email the district?") == "with_question"


def test_classification_is_per_table(monkeypatch):
    _classify(monkeypatch, {("tips", "Notes"): "content_free_text",
                            ("other", "Notes"): "metadata"})
    assert chunk_mode("tips", "Notes") == "answer_only"
    assert chunk_mode("other", "Notes") == "with_question"


# ----- unchanged behaviour ---------------------------------------------------


def test_blank_values_are_still_skipped(monkeypatch):
    _classify(monkeypatch, {("tips", "What happened?"): "content_free_text"})
    chunks, _ = chunk_document(_doc({"What happened?": "[null]"}))
    assert chunks == []


def test_parent_row_keeps_every_field(monkeypatch):
    _classify(monkeypatch, {("tips", "Your name"): "contact_or_id"})
    _, parent = chunk_document(_doc({"Your name": "Jane Doe", "What happened?": "A story."}))
    assert parent.fields["Your name"] == "Jane Doe"


# ----- a suggestion must never drop a column ---------------------------------


def test_a_classification_cannot_create_a_public_exclusion(monkeypatch):
    _classify(monkeypatch, {("tips", "Your name"): "contact_or_id"})
    assert chunk_mode("tips", "Your name") == "with_question"


def test_a_configured_policy_module_controls_the_mode(monkeypatch):
    import sys
    import types

    module = types.ModuleType("test_chunk_policy")
    module.chunk_mode = lambda source_file, column_name: "skip"
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(chunker.config, "CHUNK_POLICY_MODULE", module.__name__)
    chunker._load_chunk_policy.cache_clear()

    assert chunk_mode("tips", "email") == "skip"


def test_a_policy_module_must_return_a_known_mode(monkeypatch):
    import sys
    import types

    module = types.ModuleType("test_bad_chunk_policy")
    module.chunk_mode = lambda source_file, column_name: "maybe"
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(chunker.config, "CHUNK_POLICY_MODULE", module.__name__)
    chunker._load_chunk_policy.cache_clear()

    with pytest.raises(ValueError, match="invalid mode"):
        chunk_mode("tips", "email")
