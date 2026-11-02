"""Every strategy hands the UI and the export the same SearchResult shape."""

from re_source.vespa_strategies import AtomicVespaStrategy, BM25VespaStrategy


class _TwoChunkClient:
    """One parent, four matching chunks, and a full row to hang off it."""

    def __init__(self, score_feature):
        self.score_feature = score_feature

    def query(self, **kwargs):
        if "keyword_text" in kwargs.get("yql", ""):
            return {"root": {"children": [
                {"relevance": 0.8,
                 "fields": {"parent_id": "tips__recA",
                            "source_file": "tips_2024",
                            "full_row": {"Body": "mold in the whole row"},
                            "matchfeatures": {"bm25(keyword_text)": 0.8}}}
            ]}}
        if "from parent" in kwargs.get("yql", ""):
            return {"root": {"children": [
                {"fields": {"parent_id": "tips__recA",
                            "full_row": {"Body": "the whole row"}}}
            ]}}
        return {"root": {"children": [
            {
                "relevance": 0.1,
                "fields": {
                    "chunk_id": f"tips__recA_c{i}",
                    "parent_id": "tips__recA",
                    "column_name": f"Q{i}",
                    "column_key": f"c{i}",
                    "text": f"answer {i}",
                    "source_file": "tips_2024",
                    "matchfeatures": {self.score_feature: 0.9 - i / 100},
                },
            }
            for i in range(4)
        ]}}

    def get(self, schema, doc_id):
        return {"full_row": {"Body": "the whole row"}}


def _atomic():
    return AtomicVespaStrategy(_TwoChunkClient("secondPhase")).execute("mold")


def _bm25():
    return BM25VespaStrategy(_TwoChunkClient("bm25(keyword_text)")).execute("mold")


def test_atomic_result_carries_the_payload_the_ui_reads():
    (r,) = _atomic().results
    assert set(r.payload) == {"parent_id", "matching_chunks", "num_matching_chunks"}
    assert r.payload["parent_id"] == "tips__recA"
    assert r.payload["num_matching_chunks"] == 4


def test_bm25_result_carries_the_same_payload():
    (r,) = _bm25().results
    assert set(r.payload) == {
        "parent_id", "matching_chunks", "num_matching_chunks", "highlights"
    }


def test_the_doc_id_is_the_parent_id():
    assert _atomic().results[0].doc_id == "tips__recA"
    assert _bm25().results[0].doc_id == "tips__recA"


def test_the_display_text_is_the_first_three_chunks():
    assert _atomic().results[0].text == "answer 0\nanswer 1\nanswer 2"
    assert _bm25().results[0].text == "mold in the whole row"


def test_the_fields_are_the_parent_full_row():
    assert _atomic().results[0].fields == {"Body": "the whole row"}
    assert _bm25().results[0].fields == {"Body": "mold in the whole row"}


def test_the_score_is_the_best_matching_chunks():
    assert _atomic().results[0].score == 0.9
    assert _bm25().results[0].score == 0.8


def test_a_matching_chunk_carries_its_column_and_score():
    chunk = _atomic().results[0].payload["matching_chunks"][0]
    assert set(chunk) == {"chunk_id", "column_name", "column_key", "text", "score"}
    assert chunk["column_name"] == "Q0"
    assert chunk["score"] == 0.9


def test_keyword_results_keep_the_shared_result_interface():
    result = _bm25().results[0]
    assert result.doc_id == "tips__recA"
    assert result.source_file == "tips_2024"
    assert result.payload["matching_chunks"][0]["column_name"] == "Body"


# ----- what a strategy reports ----------------------------------------------


class _EmptyClient:
    def query(self, **kwargs):
        return {"root": {"children": []}}

    def get(self, schema, doc_id):
        raise AssertionError("a search with no hits fetched a document")


def test_a_search_with_no_hits_returns_nothing_and_fetches_nothing():
    result = AtomicVespaStrategy(_EmptyClient()).execute("nothing matches this")
    assert result.results == []


def test_an_empty_search_reports_the_full_metadata_shape():
    result = AtomicVespaStrategy(_EmptyClient()).execute("nothing matches this")
    assert result.metadata == {"num_candidates": 0, "cutoff": 0, "num_parents": 0}


def test_atomic_reports_what_retrieval_found_and_what_survived():
    result = _atomic()
    assert result.metadata["num_candidates"] == 4
    assert result.metadata["num_parents"] == 1


def test_bm25_reports_its_own_metadata_keys():
    assert set(_bm25().metadata) == {"num_parents"}
    assert set(_atomic().metadata) == {"num_candidates", "cutoff", "num_parents"}
