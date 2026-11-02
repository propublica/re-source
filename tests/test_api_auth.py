"""Okta bearer-token gate on the search API."""

import time

import pytest
from fastapi.testclient import TestClient
from joserfc import jwt
from joserfc.jwk import KeySet, OctKey, RSAKey

import api.auth
from api.main import app

client = TestClient(app)

ISSUER = "https://example.okta.com/oauth2/default"
AUDIENCE = "api://tips-search"

_KEY = RSAKey.generate_key(2048, parameters={"kid": "test-key-1"})


@pytest.fixture(autouse=True)
def okta(monkeypatch):
    """Auth on, pointed at our fake issuer, with the JWKS served locally."""
    monkeypatch.setattr(api.auth, "REQUIRE_API_AUTH", True)
    monkeypatch.setattr(api.auth, "OKTA_ISSUER", ISSUER)
    monkeypatch.setattr(api.auth, "OKTA_AUDIENCE", AUDIENCE)

    keyset = KeySet([_KEY])
    monkeypatch.setattr(api.auth, "_signing_keys", lambda force_refresh=False: keyset)


def _token(key=_KEY, alg="RS256", **overrides) -> str:
    claims = {
        "iss": ISSUER,
        "aud": AUDIENCE,
        "sub": "00u1abcdef",
        "email": "reporter@example.com",
        "scp": ["re-source:search"],
        "exp": int(time.time()) + 300,
        "iat": int(time.time()),
        **overrides,
    }
    return jwt.encode({"alg": alg, "kid": "test-key-1"}, claims, key)


def _search(token: str | None):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return client.post(
        "/search", json={"query": "x", "strategy": "Nope"}, headers=headers
    )


# ----- the gate itself -------------------------------------------------------

def test_health_needs_no_token():
    assert client.get("/health").status_code == 200


def test_search_without_a_token_is_401():
    r = _search(None)
    assert r.status_code == 401
    assert r.headers["www-authenticate"] == "Bearer"


def test_strategies_also_requires_a_token():
    assert client.get("/strategies").status_code == 401


def test_valid_token_reaches_the_endpoint():
    assert _search(_token()).status_code == 400


def test_token_without_search_scope_is_403():
    assert _search(_token(scp=[])).status_code == 403


def test_malformed_authorization_header_is_401():
    r = client.post(
        "/search", json={"query": "x", "strategy": "Content"},
        headers={"Authorization": _token()},  # no "Bearer " prefix
    )
    assert r.status_code == 401


# ----- claim validation ------------------------------------------------------

def test_expired_token_is_401():
    assert _search(_token(exp=int(time.time()) - 60)).status_code == 401


def test_wrong_audience_is_401():
    assert _search(_token(aud="api://some-other-service")).status_code == 401


def test_wrong_issuer_is_401():
    assert _search(_token(iss="https://attacker.example.com")).status_code == 401


def test_token_without_exp_is_401():
    claims = {"iss": ISSUER, "aud": AUDIENCE, "sub": "00u1abcdef"}
    token = jwt.encode({"alg": "RS256", "kid": "test-key-1"}, claims, _KEY)
    assert _search(token).status_code == 401


def test_token_without_sub_is_401():
    assert _search(_token(sub=None)).status_code == 401


# ----- signature -------------------------------------------------------------

def test_token_signed_by_another_key_is_401():
    other = RSAKey.generate_key(2048, parameters={"kid": "test-key-1"})
    assert _search(_token(key=other)).status_code == 401


def test_hmac_signed_token_is_rejected():
    token = jwt.encode(
        {"alg": "HS256", "kid": "test-key-1"},
        {"iss": ISSUER, "aud": AUDIENCE, "sub": "x", "exp": int(time.time()) + 300},
        OctKey.import_key("a" * 64),
    )
    assert _search(token).status_code == 401


def test_none_alg_is_not_accepted():
    assert "none" not in [a.lower() for a in api.auth._ALLOWED_ALGORITHMS]


# ----- principal -------------------------------------------------------------

def test_principal_prefers_email_for_attribution():
    p = api.auth.verify_token(_token())
    assert p.subject == "00u1abcdef"
    assert p.user_id == "reporter@example.com"


def test_principal_falls_back_to_subject_without_email():
    p = api.auth.verify_token(_token(email=None))
    assert p.user_id == "00u1abcdef"


def test_scopes_parse_from_either_claim_shape():
    assert api.auth.verify_token(_token(scp=["tips:search"])).scopes == ("tips:search",)
    assert api.auth.verify_token(
        _token(scp=None, scope="tips:search openid")
    ).scopes == ("tips:search", "openid")


# ----- fail-closed configuration --------------------------------------------

def test_require_api_auth_only_disabled_by_exact_false(monkeypatch):
    import importlib

    for value, expected in [
        ("false", False), ("FALSE", False), (" false ", False),
        ("true", True), ("0", True), ("no", True), ("", True), ("flase", True),
    ]:
        monkeypatch.setenv("REQUIRE_API_AUTH", value)
        assert importlib.reload(api.auth).REQUIRE_API_AUTH is expected, value

    monkeypatch.delenv("REQUIRE_API_AUTH")
    assert importlib.reload(api.auth).REQUIRE_API_AUTH is True


# ----- when a JWKS refetch is worth a network round-trip ---------------------
#
# A forced refetch is a blocking outbound request taken while holding the
# module lock. Only one failure can be fixed by it — the token names a signing
# key the cached set doesn't hold — so only that one should pay for it.
# Retrying on every JoseError let an unauthenticated caller stall every
# in-flight request, and hammer Okta from this server's IP, by looping garbage.

import requests

_ROTATED = RSAKey.generate_key(2048, parameters={"kid": "rotated-key"})
# Captured before the autouse fixture replaces it, so these tests can exercise
# the real cache-and-refetch logic rather than the stub.
_REAL_SIGNING_KEYS = api.auth._signing_keys


@pytest.fixture
def jwks(okta, monkeypatch):
    """The real _signing_keys over a counted, controllable fetch."""
    state = {"fetches": 0, "sets": [KeySet([_KEY])], "error": None}

    def _fetch():
        state["fetches"] += 1
        if state["error"] is not None:
            raise state["error"]
        i = min(state["fetches"], len(state["sets"])) - 1
        return state["sets"][i]

    monkeypatch.setattr(api.auth, "_signing_keys", _REAL_SIGNING_KEYS)
    monkeypatch.setattr(api.auth, "_fetch_keyset", _fetch)
    monkeypatch.setattr(api.auth, "_keyset", None)
    monkeypatch.setattr(api.auth, "_keyset_fetched_at", 0.0)
    return state


def _rotated_token():
    return jwt.encode(
        {"alg": "RS256", "kid": "rotated-key"},
        {"iss": ISSUER, "aud": AUDIENCE, "sub": "00u1",
         "exp": int(time.time()) + 300},
        _ROTATED,
    )


def test_a_rotated_key_still_gets_its_one_refetch(jwks):
    jwks["sets"] = [KeySet([_KEY]), KeySet([_KEY, _ROTATED])]

    assert _search(_rotated_token()).status_code != 401
    assert jwks["fetches"] == 2          # initial fill, then the refetch


def test_a_bad_signature_under_a_known_kid_does_not_refetch(jwks):
    other = RSAKey.generate_key(2048, parameters={"kid": "test-key-1"})
    token = jwt.encode(
        {"alg": "RS256", "kid": "test-key-1"},
        {"iss": ISSUER, "aud": AUDIENCE, "sub": "00u1",
         "exp": int(time.time()) + 300},
        other,
    )

    assert _search(token).status_code == 401
    assert jwks["fetches"] == 1          # the initial fill only


def test_a_token_with_no_kid_does_not_refetch(jwks):
    token = jwt.encode(
        {"alg": "RS256"},
        {"iss": ISSUER, "aud": AUDIENCE, "sub": "00u1",
         "exp": int(time.time()) + 300},
        _KEY,
    )
    _search(token)
    assert jwks["fetches"] == 1


def test_garbage_does_not_refetch(jwks):
    assert _search("not-a-jwt-at-all").status_code == 401
    assert jwks["fetches"] <= 1


def test_an_expired_token_does_not_refetch(jwks):
    assert _search(_token(exp=int(time.time()) - 10)).status_code == 401
    assert jwks["fetches"] == 1


def test_a_flood_of_bad_tokens_costs_one_fetch(jwks):
    for _ in range(25):
        _search("garbage.token.here")
    assert jwks["fetches"] <= 1


# ----- Okta being unreachable is our outage, not the caller's fault ----------

def test_an_unreachable_jwks_is_503_not_401(jwks):
    jwks["error"] = requests.ConnectionError("no route")
    assert _search(_token()).status_code == 503


def test_an_unreachable_jwks_during_the_rotation_retry_is_503(jwks):
    def _fetch():
        jwks["fetches"] += 1
        if jwks["fetches"] == 1:
            return KeySet([_KEY])
        raise requests.ConnectionError("okta down")

    api.auth._fetch_keyset = _fetch
    assert _search(_rotated_token()).status_code == 503
