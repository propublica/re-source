"""What the Author Profile strategy shows once the validator has cited evidence.

`corroborated_evidence` decides which citations survive and has its own tests.
This covers the code that *applies* that decision — the rewrite that replaces a
result's evidence chips and its body text with the cited quotes. It used to be
a mutation loop inside `validate_page`, reachable only with a VespaClient and a
live LLM, so the rule was tested thoroughly and the thing applying it was not
tested at all.

The guarantee worth pinning: the AI-written author_profile never reaches the
screen as the tipster's own prose, and an uncorroborated citation degrades to
showing less rather than to showing something false.
"""

from re_source.search_types import SearchResult
from re_source.vespa_strategies import apply_cited_evidence, cited_evidence_view


ROW = {
    "Your job": "I am a 3rd grade teacher in Dayton.",
    "What happened?": "The district cut our supply budget in half.",
    "_record_id": "rec1",
    "author_profile": "A teacher, mid-career.",
}


def _result(evidence, text="the original row text"):
    return SearchResult(
        doc_id="d1", text=text, source_file="tips", fields=dict(ROW), score=0.87,
        payload={"evidence": evidence, "llm_reason": "they teach"},
    )


# ----- the view ---------------------------------------------------------------

def test_a_corroborated_citation_is_kept_and_becomes_the_body():
    evidence, text = cited_evidence_view(
        ROW, [{"field": "Your job", "quote": "3rd grade teacher"}]
    )
    assert evidence == [{"field": "Your job", "quote": "3rd grade teacher"}]
    assert text == "Your job: 3rd grade teacher"


def test_several_citations_keep_their_order_in_the_body():
    _, text = cited_evidence_view(ROW, [
        {"field": "What happened?", "quote": "supply budget"},
        {"field": "Your job", "quote": "teacher in Dayton"},
    ])
    assert text == "What happened?: supply budget\nYour job: teacher in Dayton"


def test_an_uncorroborated_citation_yields_nothing():
    evidence, text = cited_evidence_view(
        ROW, [{"field": "Your job", "quote": "I am a nurse at St. Jude"}]
    )
    assert (evidence, text) == ([], "")


def test_the_generated_profile_cannot_become_the_body():
    """The leak this whole path exists to prevent: the model's own summary
    rendered as the tipster's words."""
    _, text = cited_evidence_view(
        ROW, [{"field": "author_profile", "quote": "A teacher, mid-career."}]
    )
    assert "mid-career" not in text
    assert text == ""


def test_no_citations_at_all_is_empty_not_an_error():
    assert cited_evidence_view(ROW, None) == ([], "")
    assert cited_evidence_view(ROW, []) == ([], "")


# ----- applying it to a result -------------------------------------------------

def test_the_result_is_rewritten_from_its_citations():
    r = _result([{"field": "Your job", "quote": "3rd grade teacher"}])
    apply_cited_evidence(r)

    assert r.payload["evidence"] == [{"field": "Your job", "quote": "3rd grade teacher"}]
    assert r.text == "Your job: 3rd grade teacher"


def test_cited_quotes_are_never_reported_as_search_matches():
    """matching_chunks means what retrieval matched. Here that was only the
    generated author_profile, so it is emptied rather than refilled with
    quotes the model chose."""
    r = _result([{"field": "Your job", "quote": "3rd grade teacher"}])
    r.payload["matching_chunks"] = [
        {"column_name": "author_profile", "text": "A teacher, mid-career."}
    ]
    apply_cited_evidence(r)

    assert r.payload["matching_chunks"] == []
    assert r.payload["num_matching_chunks"] == 0


def test_the_original_body_does_not_survive_a_rejected_citation():
    """r.text held the author_profile chunk for this strategy, so leaving it would
    show the AI summary. Blank is correct — the card falls back to llm_reason."""
    r = _result([{"field": "Your job", "quote": "invented"}], text="A teacher, mid-career.")
    apply_cited_evidence(r)

    assert r.text == ""
    assert r.payload["matching_chunks"] == []


def test_the_payload_keeps_only_the_checked_answer():
    """Both a checked and an unchecked answer to the same question used to sit
    in one payload, and the unchecked one looked authoritative."""
    r = _result([
        {"field": "Your job", "quote": "3rd grade teacher"},
        {"field": "Your job", "quote": "and I am also a nurse"},
    ])
    apply_cited_evidence(r)

    assert r.payload["evidence"] == [{"field": "Your job", "quote": "3rd grade teacher"}]


def test_the_evidence_key_stays_present_when_everything_is_rejected():
    """result_card reads its presence to choose the label — "Matching evidence"
    is still the right heading for this strategy, with nothing under it."""
    r = _result([{"field": "Your job", "quote": "invented"}])
    apply_cited_evidence(r)

    assert r.payload["evidence"] == []
    assert r.payload["evidence"] is not None


def test_a_result_with_no_payload_does_not_crash():
    r = SearchResult(doc_id="d1", text="t", source_file="tips",
                     fields=dict(ROW), score=0.1, payload=None)
    apply_cited_evidence(r)
    assert r.payload["matching_chunks"] == []


def test_llm_reason_is_left_alone():
    """The body falls back to it when evidence is empty, so the rewrite must
    not touch it."""
    r = _result([{"field": "Your job", "quote": "invented"}])
    apply_cited_evidence(r)
    assert r.payload["llm_reason"] == "they teach"
