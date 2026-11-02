"""re_source/datasets.py — dataset labels, record links, and table ownership."""

import duckdb
import pytest

import re_source.datasets as datasets


@pytest.fixture(autouse=True)
def clear_cache():
    datasets.reload_metadata()
    yield
    datasets.reload_metadata()


@pytest.fixture
def metadata(tmp_path, monkeypatch):
    """Point the catalog reader and writer at a throwaway DuckDB."""
    path = tmp_path / "tips.duckdb"
    monkeypatch.setattr(datasets, "DUCKDB_PATH", path)
    datasets.reload_metadata()

    def write(entries):
        conn = duckdb.connect(str(path))
        try:
            datasets.upsert_dataset_metadata(conn, entries)
        finally:
            conn.close()

    return write


@pytest.fixture
def read_only(monkeypatch):
    def set_patterns(*patterns):
        monkeypatch.setattr(datasets, "READ_ONLY_TABLE_PATTERNS", patterns)

    return set_patterns


def test_a_missing_database_leaves_every_dataset_plain(metadata, tmp_path, monkeypatch):
    monkeypatch.setattr(datasets, "DUCKDB_PATH", tmp_path / "nope.duckdb")
    datasets.reload_metadata()
    assert datasets.dataset_metadata() == {}
    assert datasets.dataset_label("housing_2026") == "housing_2026"
    assert datasets.record_url("housing_2026", "r1") is None
    assert datasets.is_externally_owned("housing_2026") is False


def test_a_database_without_a_catalog_leaves_every_dataset_plain(
    metadata, tmp_path, monkeypatch
):
    path = tmp_path / "empty.duckdb"
    duckdb.connect(str(path)).close()
    monkeypatch.setattr(datasets, "DUCKDB_PATH", path)
    datasets.reload_metadata()
    assert datasets.dataset_metadata() == {}


def test_a_labelled_dataset_reads_back_its_label(metadata):
    metadata({"ds1": {"label": "Housing tips"}})
    assert datasets.dataset_label("ds1") == "Housing tips"


def test_an_unlisted_dataset_falls_back_to_its_id(metadata):
    metadata({"ds1": {"label": "Housing tips"}})
    assert datasets.dataset_label("other") == "other"


@pytest.mark.parametrize("label", ["", None])
def test_a_blank_label_falls_back_rather_than_rendering_empty(metadata, label):
    metadata({"ds1": {"label": label}})
    assert datasets.dataset_label("ds1") == "ds1"


def test_a_template_becomes_a_record_link(metadata):
    metadata({"ds1": {"record_url": "https://example.com/{dataset_id}/{record_id}"}})
    assert datasets.record_url("ds1", "r1") == "https://example.com/ds1/r1"


def test_a_dataset_without_a_template_has_no_link(metadata):
    metadata({"ds1": {"label": "Housing tips"}})
    assert datasets.record_url("ds1", "r1") is None


@pytest.mark.parametrize("dataset_id,record_id", [("", "r1"), ("ds1", ""), ("ds1", None)])
def test_a_missing_id_produces_no_link(metadata, dataset_id, record_id):
    metadata({"ds1": {"record_url": "https://example.com/{record_id}"}})
    assert datasets.record_url(dataset_id, record_id) is None


def test_a_template_naming_an_unknown_field_yields_no_link(metadata):
    metadata({"ds1": {"record_url": "https://example.com/{table_id}/{record_id}"}})
    assert datasets.record_url("ds1", "r1") is None


def test_a_listed_dataset_is_externally_owned(metadata):
    metadata({"ds1": {"label": "Housing tips"}})
    assert datasets.is_externally_owned("ds1") is True


def test_ownership_ignores_case(metadata):
    metadata({"abc123": {"label": "Housing tips"}})
    assert datasets.is_externally_owned("ABC123") is True


def test_a_pattern_claims_a_table_the_catalog_has_never_seen(metadata, read_only):
    read_only(r"^ext[A-Za-z0-9]{14}$")
    assert datasets.is_externally_owned("extAbC123DeF456Gh") is True
    assert datasets.is_externally_owned("housing_2026") is False


def test_patterns_ignore_case_too(metadata, read_only):
    read_only(r"^ext[A-Za-z0-9]{14}$")
    assert datasets.is_externally_owned("EXTAbC123DeF456Gh") is True


def test_no_patterns_configured_claims_nothing(metadata, read_only):
    read_only()
    assert datasets.is_externally_owned("anything_at_all") is False


def test_catalog_upserts_preserve_an_existing_label_when_the_new_one_is_missing(metadata):
    metadata({"ds1": {"label": "First", "record_url": "https://old/{record_id}"}})
    metadata({"ds1": {"record_url": "https://new/{record_id}"}})
    assert datasets.dataset_label("ds1") == "First"
    assert datasets.record_url("ds1", "r1") == "https://new/r1"


def test_catalog_refreshes_after_an_external_database_write(metadata, tmp_path, monkeypatch):
    path = tmp_path / "external.duckdb"
    duckdb.connect(str(path)).close()
    monkeypatch.setattr(datasets, "DUCKDB_PATH", path)
    datasets.reload_metadata()
    assert datasets.dataset_label("ds1") == "ds1"

    conn = duckdb.connect(str(path))
    try:
        conn.execute(
            'CREATE TABLE dataset_catalog ('
            'dataset_id VARCHAR PRIMARY KEY, label VARCHAR, record_url VARCHAR)'
        )
        conn.execute("INSERT INTO dataset_catalog VALUES ('ds1', 'Second', NULL)")
    finally:
        conn.close()

    assert datasets.dataset_label("ds1") == "Second"


def test_an_empty_upsert_does_not_create_the_catalog(tmp_path):
    path = tmp_path / "empty.duckdb"
    conn = duckdb.connect(str(path))
    try:
        assert datasets.upsert_dataset_metadata(conn, {}) == 0
        assert conn.execute(
            "SELECT count(*) FROM information_schema.tables "
            "WHERE table_name = 'dataset_catalog'"
        ).fetchone() == (0,)
    finally:
        conn.close()


def test_deleting_catalog_entries_forgets_only_those_datasets(metadata, tmp_path):
    metadata({"ds1": {"label": "Housing tips"}, "ds2": {"label": "School tips"}})
    conn = duckdb.connect(str(tmp_path / "tips.duckdb"))
    try:
        assert datasets.delete_dataset_metadata(conn, ["ds1"]) == 1
    finally:
        conn.close()
    assert datasets.dataset_label("ds1") == "ds1"
    assert datasets.dataset_label("ds2") == "School tips"


def test_deleting_from_a_database_without_a_catalog_is_a_no_op(tmp_path):
    conn = duckdb.connect(str(tmp_path / "empty.duckdb"))
    try:
        assert datasets.delete_dataset_metadata(conn, ["ds1"]) == 0
    finally:
        conn.close()


def test_a_grouped_dataset_reads_back_its_group(metadata):
    metadata({"appA": {"label": "Callout A", "group": "wsp1"}, "appB": {"label": "B"}})

    assert datasets.dataset_group("appA") == "wsp1"
    assert datasets.dataset_group("appB") is None
    assert datasets.dataset_group("never_seen") is None


def test_catalog_upserts_preserve_an_existing_group_when_the_new_one_is_missing(metadata):
    metadata({"appA": {"label": "A", "group": "wsp1"}})
    metadata({"appA": {"label": "A renamed"}})

    assert datasets.dataset_group("appA") == "wsp1"
    assert datasets.dataset_label("appA") == "A renamed"


def test_a_catalog_from_before_groups_still_reads_and_gains_the_column(metadata, tmp_path):
    path = tmp_path / "tips.duckdb"
    conn = duckdb.connect(str(path))
    conn.execute(
        f'CREATE TABLE "{datasets.DATASET_CATALOG_TABLE}" '
        "(dataset_id VARCHAR PRIMARY KEY, label VARCHAR, record_url VARCHAR)"
    )
    conn.execute(f"INSERT INTO \"{datasets.DATASET_CATALOG_TABLE}\" VALUES ('old', 'Old', NULL)")
    conn.close()
    datasets.reload_metadata()

    assert datasets.dataset_label("old") == "Old"
    assert datasets.dataset_group("old") is None

    metadata({"new": {"label": "New", "group": "wsp2"}})

    assert datasets.dataset_group("new") == "wsp2"
    assert datasets.dataset_label("old") == "Old"
