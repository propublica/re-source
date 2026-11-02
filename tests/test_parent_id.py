"""Chunk hits resolve to a parent via the stored `parent_id` attribute, with a
fallback to deriving it from chunk_id.
"""

from re_source.vespa_chunks import _parent_id_from_fields


def test_prefers_the_stored_attribute():
    fields = {
        "parent_id": "tips__recABC123",
        "chunk_id": "tips__recABC123_body",
        "column_key": "body",
    }
    assert _parent_id_from_fields(fields) == "tips__recABC123"


def test_falls_back_when_the_field_is_absent():
    fields = {"chunk_id": "tips__recABC123_body", "column_key": "body"}
    assert _parent_id_from_fields(fields) == "tips__recABC123"


def test_falls_back_for_a_multi_part_chunk():
    fields = {"chunk_id": "tips__recABC123_body_p2", "column_key": "body"}
    assert _parent_id_from_fields(fields) == "tips__recABC123"


def test_empty_stored_value_is_not_trusted():
    fields = {"parent_id": "", "chunk_id": "tips__recABC123_body", "column_key": "body"}
    assert _parent_id_from_fields(fields) == "tips__recABC123"
