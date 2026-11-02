"""The batch sweep's strategy lists have to name strategies that exist."""

import importlib.util
from pathlib import Path

from re_source.vespa_strategies import VESPA_STRATEGIES

_spec = importlib.util.spec_from_file_location(
    "batch_search",
    Path(__file__).resolve().parent.parent / "scripts" / "batch_search.py",
)
bs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bs)


def test_every_fast_strategy_is_registered():
    unknown = [n for n in bs.FAST_STRATEGIES if n not in VESPA_STRATEGIES]
    assert unknown == [], f"--fast names strategies that don't exist: {unknown}"


def test_fast_is_not_empty():
    assert bs.FAST_STRATEGIES


def test_fast_excludes_the_llm_strategies():
    for name in bs.FAST_STRATEGIES:
        assert not getattr(VESPA_STRATEGIES[name], "paginates", False), name


# ----- --merge keys on the record, not the dataset ------------------------------
#
# source_file is a DuckDB table name — one per dataset. Keying the merge on it
# meant that once a single row from a dataset was in the CSV, every later row from
# that dataset was discarded as "already existing". A sweep over 88 datasets kept at
# most 88 rows, silently.

from pathlib import Path


def _result(doc_id, source_file="app_one"):
    return {
        "doc_id": doc_id,
        "source_file": source_file,
        "score": 0.5,
        "matching_chunks": [{"text": f"text for {doc_id}"}],
    }


def test_two_rows_from_one_base_both_survive_a_merge(tmp_path):
    out = tmp_path / "q.csv"
    bs.export_query_results([_result("recA"), _result("recB")], str(out))

    existing = bs.load_existing_results(Path(out))

    assert existing == {"recA", "recB"}


def test_a_new_row_from_an_already_present_base_is_not_discarded(tmp_path):
    out = tmp_path / "q.csv"
    bs.export_query_results([_result("recA")], str(out))

    existing = bs.load_existing_results(Path(out))
    incoming = [_result("recA"), _result("recB")]
    kept = [r for r in incoming if r.get("doc_id", "") not in existing]

    assert [r["doc_id"] for r in kept] == ["recB"]


def test_the_export_carries_the_record_so_a_reader_can_find_it(tmp_path):
    import csv

    out = tmp_path / "q.csv"
    bs.export_query_results([_result("recA")], str(out))

    with open(out, encoding="utf-8") as f:
        (row,) = list(csv.DictReader(f))

    assert row["doc_id"] == "recA"
    assert row["source_file"] == "app_one"


def test_merging_into_a_file_that_does_not_exist_yet_keeps_everything(tmp_path):
    assert bs.load_existing_results(tmp_path / "nope.csv") == set()
