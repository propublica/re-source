"""FastAPI endpoint tests. /search runs against a patched run_search so no
Vespa is required, and Okta auth is switched off so these cover endpoint
behavior only — the auth gate itself is tested in test_api_auth.py."""

import pytest
from fastapi.testclient import TestClient

import api.auth
from api.main import app
from re_source.search_types import SearchResult
from re_source.vespa_strategies import StrategyResult

client = TestClient(app)


@pytest.fixture(autouse=True)
def no_auth(monkeypatch):
    """Run these as if REQUIRE_API_AUTH=false (the local-dev mode)."""
    monkeypatch.setattr(api.auth, "REQUIRE_API_AUTH", False)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_strategies_lists_all():
    from re_source.vespa_strategies import VESPA_STRATEGIES

    r = client.get("/strategies")
    assert r.status_code == 200
    # Endpoint exposes exactly the live registry (robust to added strategies).
    assert set(r.json()["strategies"]) == set(VESPA_STRATEGIES)
    assert "Author Profile + LLM" in r.json()["strategies"]


def test_search_unknown_strategy_is_400():
    r = client.post("/search", json={"query": "x", "strategy": "Nope"})
    assert r.status_code == 400


def test_search_serializes_results(monkeypatch):
    fake = StrategyResult(
        results=[
            SearchResult(
                doc_id="rec1", text="t", source_file="app1",
                fields={"_record_id": "rec1", "Name": "Jane"}, score=0.9,
                payload={"matching_chunks": [], "llm_reason": "ok"},
            )
        ],
        elapsed_seconds=0.5,
        approach_name="Content (Vespa)",
        metadata={"before_llm": 1},
    )
    monkeypatch.setattr("api.main.run_search", lambda *a, **k: fake)

    r = client.post("/search", json={"query": "teachers", "strategy": "Content", "top_k": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["approach_name"] == "Content (Vespa)"
    assert body["elapsed_seconds"] == 0.5
    assert body["results"][0]["doc_id"] == "rec1"
    assert body["results"][0]["payload"]["llm_reason"] == "ok"


def test_search_validates_top_k_bounds():
    r = client.post("/search", json={"query": "x", "strategy": "Content", "top_k": 99999})
    assert r.status_code == 422  # exceeds le=2000


def _fake_result(*record_ids):
    return StrategyResult(
        results=[
            SearchResult(
                doc_id=rid, text="t", source_file="app1",
                fields={"_record_id": rid}, score=0.9, payload=None,
            )
            for rid in record_ids
        ],
        elapsed_seconds=0.1,
        approach_name="Content (Vespa)",
        metadata={},
    )


def test_search_records_an_audit_event(monkeypatch):
    calls = []
    monkeypatch.setattr("api.main.run_search", lambda *a, **k: _fake_result("rec1", "rec2"))
    monkeypatch.setattr("api.main.audit.log_event", lambda *a, **kw: calls.append((a, kw)))

    r = client.post("/search", json={"query": "teachers", "strategy": "Content", "top_k": 5})
    assert r.status_code == 200

    (args, kwargs), = calls
    assert args[0] == "results_viewed"
    assert kwargs["surface"] == "api"
    assert kwargs["record_ids"] == ["app1::rec1", "app1::rec2"]
    assert "query" not in kwargs
    assert kwargs["strategy"] == "Content"


def test_search_audit_failure_does_not_break_the_response(monkeypatch):
    monkeypatch.setattr("api.main.run_search", lambda *a, **k: _fake_result("rec1"))
    r = client.post("/search", json={"query": "teachers", "strategy": "Content"})
    assert r.status_code == 200
    assert r.json()["results"][0]["doc_id"] == "rec1"


def test_the_default_strategy_is_one_the_registry_holds():
    from api.schemas import SearchRequest
    from re_source.vespa_strategies import VESPA_STRATEGIES

    assert SearchRequest(query="teachers").strategy in VESPA_STRATEGIES


def test_a_request_without_a_strategy_is_served(monkeypatch):
    monkeypatch.setattr("api.main.run_search", lambda *a, **k: _fake_result("rec1"))
    r = client.post("/search", json={"query": "teachers"})
    assert r.status_code == 200


# ----- the request body's bounds ---------------------------------------------

def test_an_empty_query_is_rejected():
    assert client.post("/search", json={"query": ""}).status_code == 422


def test_an_oversized_query_is_rejected():
    """The query is re-sent in every LLM validation batch and written verbatim
    into the audit event, so an unbounded one is multiplied before it reaches
    the provider. min_length was declared without a matching max."""
    from api.schemas import MAX_QUERY_CHARS

    r = client.post("/search", json={"query": "x" * (MAX_QUERY_CHARS + 1)})
    assert r.status_code == 422


def test_a_query_at_the_limit_is_accepted_by_the_schema():
    """The bound must not reject a realistic search; failure here should be
    about the strategy, not the length."""
    from api.schemas import MAX_QUERY_CHARS

    r = client.post(
        "/search", json={"query": "x" * MAX_QUERY_CHARS, "strategy": "Nope"}
    )
    assert r.status_code == 400          # unknown strategy, not 422
