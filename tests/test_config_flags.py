"""Security flags parsed from the environment fail closed.

An operator typo must not silently disable a control. REQUIRE_LOGIN is the
only gate on the Streamlit surface — the one serving the tip corpus — and it
used to be parsed `== "true"`, so every truthy spelling other than the literal
"true" turned it off with nothing logged. api/auth.REQUIRE_API_AUTH already
had the right shape; this pins both to it.
"""

import importlib

import re_source.config


def _reloaded(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("REQUIRE_LOGIN", raising=False)
    else:
        monkeypatch.setenv("REQUIRE_LOGIN", value)
    return importlib.reload(re_source.config).REQUIRE_LOGIN


def test_only_an_exact_false_disables_the_login_gate(monkeypatch):
    for value, expected in [
        ("false", False), ("FALSE", False), (" false ", False),
        ("true", True), ("0", True), ("no", True), ("", True),
        ("ture", True), ("1", True), ("yes", True), ("true ", True),
    ]:
        assert _reloaded(monkeypatch, value) is expected, value


def test_the_gate_is_on_when_the_variable_is_absent(monkeypatch):
    assert _reloaded(monkeypatch, None) is True


def test_it_matches_the_api_flag(monkeypatch):
    """The two doors onto the same corpus must not disagree about what
    'off' means."""
    import api.auth

    for value in ["1", "no", "ture", "", "true "]:
        monkeypatch.setenv("REQUIRE_LOGIN", value)
        monkeypatch.setenv("REQUIRE_API_AUTH", value)
        ui = importlib.reload(re_source.config).REQUIRE_LOGIN
        api_flag = importlib.reload(api.auth).REQUIRE_API_AUTH
        assert ui == api_flag is True, value


def _restore(monkeypatch):
    monkeypatch.delenv("REQUIRE_LOGIN", raising=False)
    importlib.reload(re_source.config)


def teardown_module():
    """Leave the module as the rest of the suite expects to find it."""
    importlib.reload(re_source.config)
