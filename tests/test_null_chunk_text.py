"""A chunk doc with a null/absent `text` field must not crash retrieval.

Stale docs (fed before the rec-id migration) can come back from Vespa without
a text field. That used to put None into the chunk dict, and the "\n".join()
building a result's display text raised TypeError — taking down every content
lane at once and leaving the search with zero results.

The guard itself is tested in tests/test_chunk_readout.py, where it lives now.
This is the end-to-end check that a real strategy survives such a doc: the
guard used to be written out once per strategy, so it needed testing once per
strategy too.
"""

from re_source.vespa_strategies import AtomicVespaStrategy


class _StubClient:
    """Returns one hit whose `text` field is missing entirely."""

    def query(self, **_kwargs):
        return {
            "root": {
                "children": [
                    {
                        "relevance": 0.5,
                        "fields": {
                            "chunk_id": "recABC123__body__0",
                            "column_name": "Body",
                            "column_key": "body",
                            "source_file": "tips.csv",
                            "matchfeatures": {"secondPhase": 0.5},
                            # no "text" key at all
                        },
                    }
                ]
            }
        }

    def get(self, _schema, _doc_id):
        return {"full_row": {"Body": None}}


def test_a_search_survives_a_null_chunk_text():
    strategy = AtomicVespaStrategy(_StubClient())
    result = strategy.execute("methane emissions")
    assert [r.text for r in result.results] == [""]
