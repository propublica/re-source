"""The prompts say to the model what we think they say."""

import re

import pytest

import re_source.llm_utils as llm
import re_source.query_rewrite_cache as qrc


@pytest.fixture
def sent(monkeypatch, tmp_path):
    """Capture every prompt that reaches the model instead of sending it."""
    prompts = []
    monkeypatch.setattr(qrc, "CACHE_DIR", tmp_path / "cache")

    def _capture(**kwargs):
        for message in kwargs.get("messages", ()):
            prompts.append(message.get("content", ""))
        return '{"results": [], "expanded_query": "x"}'

    monkeypatch.setattr(llm, "_llm_generate", _capture)
    return prompts


def _escape_leaks(text: str) -> list[str]:
    """Places where an f-string escape leaked into a plain string."""
    return re.findall(r"\{\{", text)


# ----- the bug, at the level it can recur ------------------------------------

def test_no_prompt_shows_the_model_an_escaped_brace(sent):
    llm.generate_hyde_document("teachers")
    llm.expand_query("teachers")
    llm.validate_relevance_batch("teachers", ["a document"])

    assert sent, "nothing was captured — the seam moved"
    for prompt in sent:
        assert _escape_leaks(prompt) == [], prompt


def test_the_expander_asks_for_a_shape_it_can_parse_back(sent):
    llm.expand_query("teachers")

    (prompt,) = [p for p in sent if "expanded_query" in p]

    assert 'Return JSON: {"expanded_query": "..."}' in prompt


# ----- the specs the batch validators send -----------------------------------

@pytest.mark.parametrize("spec_name", [
    "_BATCH_JSON_SPEC",
    "_RELEVANCE_JSON_SPEC",
    "_AUTHOR_PROFILE_JSON_SPEC",
])
def test_every_json_spec_is_free_of_escaped_braces(spec_name):
    spec = getattr(llm, spec_name, None)
    if spec is None:
        pytest.skip(f"{spec_name} no longer exists")
    assert _escape_leaks(spec) == [], spec


@pytest.mark.parametrize("spec_name", [
    "_BATCH_JSON_SPEC",
    "_RELEVANCE_JSON_SPEC",
    "_AUTHOR_PROFILE_JSON_SPEC",
])
def test_every_batch_spec_asks_for_the_results_list_the_parser_reads(spec_name):
    spec = getattr(llm, spec_name, None)
    if spec is None:
        pytest.skip(f"{spec_name} no longer exists")
    assert '"results"' in spec
    assert '"index"' in spec
    assert '"relevant"' in spec


def test_the_relevance_spec_asks_for_the_highlights_the_screen_marks():
    assert '"highlights"' in llm._RELEVANCE_JSON_SPEC


def test_the_author_spec_asks_for_the_evidence_the_card_shows():
    assert '"evidence"' in llm._AUTHOR_PROFILE_JSON_SPEC


def test_no_spec_asks_for_a_field_nothing_reads():
    for name in ("_BATCH_JSON_SPEC", "_RELEVANCE_JSON_SPEC",
                 "_AUTHOR_PROFILE_JSON_SPEC"):
        assert "confidence" not in getattr(llm, name)


# ----- submissions are labelled as untrusted data ----------------------------
#
# Form answers written by anonymous members of the public share one prompt with
# the instructions that decide what a journalist sees, so a submission can
# address the model about the submissions beside it. `--- Document N ---`
# separated them and said nothing about trust. This is framing, not a control —
# the guarantees are downstream in _parse_batch_response and
# corroborated_evidence — but framing the model is measurably likelier to honor.

def test_the_document_block_is_marked_untrusted(sent):
    llm.validate_relevance_batch("teachers", ["a document"])

    (prompt,) = [p for p in sent if "a document" in p]
    assert llm.UNTRUSTED_OPEN in prompt
    assert llm.UNTRUSTED_CLOSE in prompt


def test_the_submission_sits_inside_the_markers(sent):
    llm.validate_relevance_batch("teachers", ["SUBMISSION TEXT"])

    (prompt,) = [p for p in sent if "SUBMISSION TEXT" in p]
    start = prompt.index(llm.UNTRUSTED_OPEN)
    end = prompt.index(llm.UNTRUSTED_CLOSE)
    assert start < prompt.index("SUBMISSION TEXT") < end


def test_the_rules_sit_outside_the_markers(sent):
    """The JSON spec and the judging rules must not be inside the block the
    prompt has just described as untrusted."""
    llm.validate_relevance_batch("teachers", ["a document"])

    (prompt,) = [p for p in sent if "a document" in p]
    end = prompt.index(llm.UNTRUSTED_CLOSE)
    assert prompt.index('"results"') > end


def test_the_framing_precedes_the_untrusted_text(sent):
    """Said before the model reads any of it, not after."""
    llm.validate_relevance_batch("teachers", ["a document"])

    (prompt,) = [p for p in sent if "a document" in p]
    assert prompt.index("never instructions") < prompt.index(llm.UNTRUSTED_OPEN)


@pytest.mark.parametrize("marker", ["UNTRUSTED_OPEN", "UNTRUSTED_CLOSE"])
def test_the_markers_are_not_plausible_form_answers(marker):
    """A submission that could reproduce the closing marker could end the block
    early and have what follows read as rules."""
    value = getattr(llm, marker)
    assert value.startswith("<<<") and value.endswith(">>>")
    assert value.upper() == value


def test_every_public_batch_validator_labels_its_documents(sent):
    """Both public validators share the untrusted-document framing."""
    llm.validate_relevance_batch("q", ["doc a"])
    llm.validate_author_profile_batch("q", ["doc b"])

    for needle in ("doc a", "doc b"):
        (prompt,) = [p for p in sent if needle in p]
        assert llm.UNTRUSTED_OPEN in prompt, needle


def test_the_enrichment_prompt_labels_the_submission(monkeypatch):
    """The other door tip text reaches a model through."""
    from re_source import enrich

    captured = {}

    def _capture(**kwargs):
        captured["messages"] = kwargs["messages"]
        return '{"author_profile": "x"}'

    monkeypatch.setattr(enrich, "_llm_generate", _capture)
    enrich.generate_author_profile("I am a nurse.")

    roles = {m["role"]: m["content"] for m in captured["messages"]}
    assert llm.UNTRUSTED_OPEN in roles["user"]
    assert "I am a nurse." in roles["user"]
    # The trusted half stays trusted: no submission text in the system role.
    assert "I am a nurse." not in roles["system"]


def test_prompts_address_a_newsroom(monkeypatch, sent):
    from re_source import enrich

    monkeypatch.setattr(
        enrich,
        "_llm_generate",
        lambda **kwargs: sent.extend(
            message["content"] for message in kwargs["messages"]
        ) or '{"author_profile": ""}',
    )

    llm.generate_hyde_document("housing")
    enrich.generate_author_profile("submission")

    assert len(sent) == 3
    assert all("newsroom" in text for text in sent[:2])


# ----- what entered the context is recorded ----------------------------------
#
# The corpus is public-authored, so a submission engineered to dominate a batch
# is worth being able to see after the fact. Sizes and counts only — which
# records were disclosed belongs in the audit log, which has retention rules
# and access controls the application log does not.

def test_the_context_size_is_logged(sent, caplog):
    import logging

    with caplog.at_level(logging.INFO, logger="re_source.llm_utils"):
        llm.validate_relevance_batch("q", ["aaa", "bbbb"])

    (line,) = [r.getMessage() for r in caplog.records if "llm context" in r.getMessage()]
    assert "2 documents" in line
    assert "7 chars" in line


def test_truncated_documents_are_counted(sent, caplog):
    """The flooding signal: one document hitting the cap is visible."""
    import logging

    with caplog.at_level(logging.INFO, logger="re_source.llm_utils"):
        llm.validate_relevance_batch("q", ["x" * 5000, "short"], max_doc_chars=100)

    (line,) = [r.getMessage() for r in caplog.records if "llm context" in r.getMessage()]
    assert "1 truncated" in line


def test_no_documents_logs_nothing(sent, caplog):
    import logging

    with caplog.at_level(logging.INFO, logger="re_source.llm_utils"):
        llm._build_doc_list([], 100)

    assert not [r for r in caplog.records if "llm context" in r.getMessage()]


def test_the_log_line_carries_no_submission_text(sent, caplog):
    """It is an operational signal, not a second copy of the corpus."""
    import logging

    with caplog.at_level(logging.INFO, logger="re_source.llm_utils"):
        llm.validate_relevance_batch("q", ["I am a nurse at St. Jude"])

    for record in caplog.records:
        assert "nurse" not in record.getMessage()
        assert "St. Jude" not in record.getMessage()
