"""Chunks are sized in tokens, so neither embedder truncates them."""

import pytest

from re_source import chunker
from re_source.chunker import (
    MAX_CHUNK_TOKENS,
    TOKEN_MARGIN,
    chunk_document,
    count_tokens,
    split_by_tokens,
)
from re_source.db_loader import Document

BUDGET = MAX_CHUNK_TOKENS - TOKEN_MARGIN


@pytest.fixture(autouse=True)
def _no_inventory(monkeypatch):
    monkeypatch.setattr(chunker, "_field_classes_cache", {})


def _doc(fields):
    return Document(id="tips__rec1", source_file="tips", fields=fields,
                    content_hash="h", text="")


# ----- the splitter ----------------------------------------------------------


def test_short_text_is_not_split():
    assert split_by_tokens("a short answer", BUDGET) == ["a short answer"]


def test_every_part_fits_the_budget():
    parts = split_by_tokens("word " * 3000, 200)
    assert len(parts) > 1
    assert all(count_tokens(p) <= 200 for p in parts)


def test_parts_preserve_the_original_text():
    text = "The Fukushima Daiichi plant. " * 200
    parts = split_by_tokens(text, 100)
    assert all(p.strip() in text for p in parts)
    assert "Fukushima Daiichi" in parts[0]


def test_a_nonsense_budget_does_not_hang():
    assert split_by_tokens("some text", 0) == ["some text"]


# ----- how chunk_document uses it --------------------------------------------


def test_a_long_answer_is_split_and_every_chunk_fits(monkeypatch):
    monkeypatch.setattr(chunker, "_field_classes_cache",
                        {("tips", "What happened?"): "content_free_text"})
    chunks, _ = chunk_document(_doc({"What happened?": "word " * 4000}))
    assert len(chunks) > 1
    assert all(count_tokens(c.text) <= BUDGET for c in chunks)


def test_a_split_answer_never_repeats_the_question(monkeypatch):
    monkeypatch.setattr(chunker, "_field_classes_cache",
                        {("tips", "Describe the incident"): "content_closed_ended"})
    chunks, _ = chunk_document(_doc({"Describe the incident": "word " * 4000}))
    assert len(chunks) > 1
    assert not any(c.text.startswith("Describe the incident") for c in chunks)


def test_a_short_answer_still_gets_its_question(monkeypatch):
    monkeypatch.setattr(chunker, "_field_classes_cache",
                        {("tips", "Are you a teacher?"): "content_closed_ended"})
    chunks, _ = chunk_document(_doc({"Are you a teacher?": "No"}))
    assert [c.text for c in chunks] == ["Are you a teacher? No"]


def test_multipart_chunks_get_distinct_ids(monkeypatch):
    monkeypatch.setattr(chunker, "_field_classes_cache",
                        {("tips", "What happened?"): "content_free_text"})
    chunks, _ = chunk_document(_doc({"What happened?": "word " * 4000}))
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))
    assert all("_p" in i for i in ids)


# ----- fallback when the tokenizer is unavailable ----------------------------


def test_falls_back_to_characters_without_a_tokenizer(monkeypatch):
    monkeypatch.setattr(chunker, "_tokenizer_cache", None)
    monkeypatch.setattr(chunker, "_tokenizer_loaded", True)
    parts = split_by_tokens("word " * 3000, 200)
    assert len(parts) > 1
    assert all(len(p) <= 200 * chunker.FALLBACK_CHARS_PER_TOKEN for p in parts)
