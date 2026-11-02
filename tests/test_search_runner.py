"""run_search dispatch tests (no Vespa needed) + one gated integration test."""

import os
import json
import urllib.request
from unittest.mock import MagicMock

import pytest

import re_source.search_runner as runner


def test_unknown_strategy_raises():
    with pytest.raises(ValueError):
        runner.run_search("Does Not Exist", "teachers")


def test_dispatches_with_client_and_args(monkeypatch):
    sentinel_result = object()
    strat = MagicMock()
    strat.execute.return_value = sentinel_result
    fake_cls = MagicMock(return_value=strat)

    monkeypatch.setitem(runner.VESPA_STRATEGIES, "Content", fake_cls)
    monkeypatch.setattr(runner, "get_vespa_client", lambda: "CLIENT")

    out = runner.run_search("Content", "teachers", top_k=7)

    assert out is sentinel_result
    fake_cls.assert_called_once_with("CLIENT")
    strat.execute.assert_called_once_with(query="teachers", top_k=7)


def _vespa_up() -> bool:
    endpoint = os.getenv("VESPA_ENDPOINT", "http://localhost:8080")
    try:
        urllib.request.urlopen(f"{endpoint}/ApplicationStatus", timeout=2)
        return True
    except Exception:
        return False


def _keyword_schema_up() -> bool:
    endpoint = os.getenv("VESPA_ENDPOINT", "http://localhost:8080")
    body = json.dumps({
        "yql": 'select * from parent where keyword_text contains "schema-probe"',
        "ranking.profile": "keyword",
        "hits": 0,
    }).encode()
    request = urllib.request.Request(
        f"{endpoint}/search/",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        urllib.request.urlopen(request, timeout=2)
        return True
    except Exception:
        return False


@pytest.mark.integration
@pytest.mark.skipif(not _vespa_up(), reason="Vespa not running")
def test_real_vespa_keyword_search():
    if not _keyword_schema_up():
        pytest.skip("running Vespa does not have this checkout's keyword schema")
    result = runner.run_search("BM25", "teachers", top_k=5)
    assert hasattr(result, "results")
    assert result.approach_name
