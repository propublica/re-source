"""VESPA_STRATEGIES registry + get_vespa_client singleton live in re_source/ and are
usable without a running Vespa."""

from unittest.mock import patch

import re_source.vespa_client as vc
from re_source.vespa_strategies import VESPA_STRATEGIES

# Core strategies that must always be registered. Kept as a subset (not an exact
# set) so adding new strategies doesn't break this test.
CORE = {"Content", "BM25", "Author Profile + LLM"}


def test_registry_has_core_strategies():
    assert CORE <= set(VESPA_STRATEGIES)


def test_every_strategy_constructs_with_client():
    fake_client = object()
    for name, cls in VESPA_STRATEGIES.items():
        strat = cls(fake_client)
        assert hasattr(strat, "execute"), f"{name} has no execute()"


def test_get_vespa_client_is_a_singleton():
    # Patch VespaClient so no real connection is attempted, and reset the cache.
    with patch.object(vc, "VespaClient") as MockClient:
        vc._client = None
        a = vc.get_vespa_client()
        b = vc.get_vespa_client()
        assert a is b
        MockClient.assert_called_once()
    vc._client = None  # leave global clean for other tests


def test_content_lanes_exclude_bm25():
    from re_source.vespa_strategies import CombinedVespaStrategy

    assert set(CombinedVespaStrategy(None)._lanes) == {
        "atomic", "hyde", "query_expansion", "atomic_decomp",
    }


def test_bm25_remains_a_selectable_strategy():
    from re_source.vespa_strategies import VESPA_STRATEGIES, BM25VespaStrategy

    assert VESPA_STRATEGIES["BM25"] is BM25VespaStrategy


# ----- a wipe must actually empty the schema ---------------------------------


class _FakeVespa:
    """delete_all that leaves a residue, like pyvespa's does when per-slice
    errors stay under its 10% tolerance."""

    def __init__(self, counts):
        self.counts = list(counts)
        self.deletes = 0

    def delete_all(self, schema, slices=8):
        self.deletes += 1

    def count(self, schema):
        return self.counts.pop(0) if self.counts else 0


def _client(counts):
    from re_source.vespa_client import VespaClient
    c = VespaClient.__new__(VespaClient)
    fake = _FakeVespa(counts)
    c.delete_all = fake.delete_all
    c.count = fake.count
    return c, fake


def test_a_wipe_that_leaves_residue_is_retried():
    from re_source.vespa_client import VespaClient
    c, fake = _client([36_000, 900, 0])
    assert VespaClient.delete_all_verified(c, "chunk") == 0
    assert fake.deletes == 3


def test_a_stalled_wipe_raises_rather_than_feeding_over_it():
    import pytest as _pytest
    from re_source.vespa_client import VespaClient
    c, _ = _client([36_000, 36_000])
    with _pytest.raises(RuntimeError, match="stalled"):
        VespaClient.delete_all_verified(c, "chunk")


def test_an_already_empty_schema_deletes_once():
    from re_source.vespa_client import VespaClient
    c, fake = _client([0])
    assert VespaClient.delete_all_verified(c, "chunk") == 0
    assert fake.deletes == 1
