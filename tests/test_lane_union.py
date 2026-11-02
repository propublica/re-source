"""How several lanes' results become one ranked pool."""

from re_source.lane_union import is_contentless, rank
from re_source.search_types import SearchResult

COMPARABLE = ("atomic", "hyde", "query_expansion")


def _r(doc_id, score, *, chunks=1, text="an answer", column="What happened?"):
    return SearchResult(
        doc_id=doc_id,
        text=text,
        source_file="tips_2024",
        fields={"_record_id": doc_id},
        score=score,
        payload={"matching_chunks": [
            {"chunk_id": f"{doc_id}_c{i}", "column_name": column,
             "text": f"{column} {text}"}
            for i in range(chunks)
        ]},
    )


# ----- which copy of a row survives ------------------------------------------

def test_a_row_found_by_two_lanes_appears_once():
    pool = rank({"atomic": [_r("p1", 0.9)], "hyde": [_r("p1", 0.8)]},
                comparable_lanes=COMPARABLE)
    assert [r.doc_id for r in pool.results] == ["p1"]


def test_the_copy_with_the_most_chunks_wins():
    thin, rich = _r("p1", 0.9, chunks=1), _r("p1", 0.5, chunks=3)
    pool = rank({"atomic": [thin], "hyde": [rich]}, comparable_lanes=COMPARABLE)
    assert len(pool.results[0].payload["matching_chunks"]) == 3


def test_a_row_records_every_lane_that_found_it():
    pool = rank({"atomic": [_r("p1", 0.9)], "hyde": [_r("p1", 0.8)]},
                comparable_lanes=COMPARABLE)
    assert pool.results[0].payload["lanes"] == ["atomic", "hyde"]


# ----- which score is comparable ---------------------------------------------

def test_the_best_comparable_score_wins():
    pool = rank({"atomic": [_r("p1", 0.4)], "hyde": [_r("p1", 0.9)]},
                comparable_lanes=COMPARABLE)
    assert pool.results[0].score == 0.9


def test_a_non_comparable_lanes_score_is_never_used_as_is():
    pool = rank({"bm25": [_r("p1", 42.0)]}, comparable_lanes=COMPARABLE)
    assert pool.results[0].score != 42.0


def test_rows_come_back_best_first():
    pool = rank({"atomic": [_r("p1", 0.4), _r("p2", 0.9)]},
                comparable_lanes=COMPARABLE)
    assert [r.doc_id for r in pool.results] == ["p2", "p1"]


# ----- the tail --------------------------------------------------------------

def test_a_row_only_a_non_comparable_lane_found_is_rescored():
    pool = rank(
        {"atomic": [_r("p1", 0.9)], "bm25": [_r("p2", 42.0)]},
        comparable_lanes=COMPARABLE,
        rescore=lambda ids: {"p2": 0.95},
    )
    assert [r.doc_id for r in pool.results] == ["p2", "p1"]
    assert pool.results[0].score == 0.95


def test_only_the_tail_is_sent_for_rescoring():
    asked = []
    rank(
        {"atomic": [_r("p1", 0.9)], "bm25": [_r("p2", 42.0)]},
        comparable_lanes=COMPARABLE,
        rescore=lambda ids: asked.extend(ids) or {},
    )
    assert all(i.startswith("p2") for i in asked), asked


def test_nothing_is_rescored_when_every_row_is_comparable():
    called = []
    rank({"atomic": [_r("p1", 0.9)]}, comparable_lanes=COMPARABLE,
         rescore=lambda ids: called.append(ids) or {})
    assert called == []


def test_a_row_the_rescore_missed_is_kept_below_the_floor():
    pool = rank(
        {"atomic": [_r("p1", 0.5)], "bm25": [_r("p2", 42.0)]},
        comparable_lanes=COMPARABLE,
        rescore=lambda ids: {},
    )
    assert [r.doc_id for r in pool.results] == ["p1", "p2"]
    assert pool.results[-1].score < 0.5


def test_a_failed_rescore_does_not_lose_the_tail():
    pool = rank(
        {"bm25": [_r("p1", 42.0), _r("p2", 41.0)]},
        comparable_lanes=COMPARABLE,
        rescore=lambda ids: None,
    )
    assert len(pool.results) == 2


def test_no_rescore_at_all_still_ranks():
    pool = rank({"bm25": [_r("p1", 42.0)]}, comparable_lanes=COMPARABLE)
    assert len(pool.results) == 1


# ----- what is not worth showing ---------------------------------------------

def test_a_question_label_over_a_blank_value_is_dropped():
    noise = _r("p1", 0.9, text="[null]")
    assert is_contentless(noise) is True

    pool = rank({"atomic": [noise]}, comparable_lanes=COMPARABLE)
    assert pool.results == []
    assert pool.metadata["contentless_dropped"] == 1


def test_one_real_answer_keeps_the_row():
    r = _r("p1", 0.9, chunks=0)
    r.payload["matching_chunks"] = [
        {"chunk_id": "a", "column_name": "Q?", "text": "Q? [null]"},
        {"chunk_id": "b", "column_name": "Q2?", "text": "Q2? a real answer"},
    ]
    assert is_contentless(r) is False


def test_a_row_with_no_chunks_and_no_text_is_dropped():
    empty = _r("p1", 0.9, chunks=0, text="")
    assert is_contentless(empty) is True


# ----- what it reports --------------------------------------------------------

def test_the_metadata_accounts_for_the_whole_pool():
    pool = rank(
        {"atomic": [_r("p1", 0.9), _r("p3", 0.9, text="[null]")],
         "bm25": [_r("p2", 42.0)]},
        comparable_lanes=COMPARABLE,
        rescore=lambda ids: {"p2": 0.5},
    )
    assert pool.metadata["union_size"] == 3
    assert pool.metadata["per_lane_counts"] == {"atomic": 2, "bm25": 1}
    assert pool.metadata["tail_size"] == 1
    assert pool.metadata["tail_reranked"] == 1
    assert pool.metadata["contentless_dropped"] == 1
    assert pool.metadata["pool_size"] == 2


def test_no_lanes_found_anything():
    pool = rank({"atomic": [], "bm25": []}, comparable_lanes=COMPARABLE)
    assert pool.results == []
    assert pool.metadata["union_size"] == 0
