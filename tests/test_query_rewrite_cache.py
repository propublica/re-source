"""Persistent cache for LLM query rewrites."""

import json

import pytest

from re_source import query_rewrite_cache as qrc


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(qrc, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.delenv("QUERY_REWRITE_CACHE", raising=False)


def _counting(value):
    calls = []

    def compute():
        calls.append(1)
        return value

    return compute, calls


def test_second_call_does_not_recompute():
    compute, calls = _counting("hypothetical doc")
    a = qrc.cached_rewrite("hyde", "prompt", "local:m", compute)
    b = qrc.cached_rewrite("hyde", "prompt", "local:m", compute)
    assert a == b == "hypothetical doc"
    assert len(calls) == 1


def test_editing_the_prompt_invalidates():
    compute, calls = _counting("x")
    qrc.cached_rewrite("hyde", "prompt v1", "local:m", compute)
    qrc.cached_rewrite("hyde", "prompt v2", "local:m", compute)
    assert len(calls) == 2


def test_switching_model_invalidates():
    compute, calls = _counting("x")
    qrc.cached_rewrite("hyde", "prompt", "local:qwen", compute)
    qrc.cached_rewrite("hyde", "prompt", "private:model", compute)
    assert len(calls) == 2


def test_kinds_do_not_collide():
    qrc.cached_rewrite("hyde", "p", "m", lambda: "hyde result")
    assert qrc.cached_rewrite("expand_query", "p", "m", lambda: "expand result") == "expand result"


def test_lists_round_trip():
    qrc.cached_rewrite("decompose", "p", "m", lambda: ["a", "b"])
    assert qrc.cached_rewrite("decompose", "p", "m", lambda: ["should not run"]) == ["a", "b"]


def test_none_is_not_cached():
    calls = []

    def failing():
        calls.append(1)
        return None

    assert qrc.cached_rewrite("decompose", "p", "m", failing) is None
    assert qrc.cached_rewrite("decompose", "p", "m", failing) is None
    assert len(calls) == 2


def test_disabled_by_env(monkeypatch):
    monkeypatch.setenv("QUERY_REWRITE_CACHE", "0")
    compute, calls = _counting("x")
    qrc.cached_rewrite("hyde", "p", "m", compute)
    qrc.cached_rewrite("hyde", "p", "m", compute)
    assert len(calls) == 2


def test_corrupt_entry_falls_through(monkeypatch):
    qrc.cached_rewrite("hyde", "p", "m", lambda: "original")
    entry = next(qrc.CACHE_DIR.glob("*.json"))
    entry.write_text("{ not valid json", encoding="utf-8")
    assert qrc.cached_rewrite("hyde", "p", "m", lambda: "recomputed") == "recomputed"


def test_unwritable_cache_dir_still_returns(monkeypatch):
    monkeypatch.setattr(qrc, "CACHE_DIR", qrc.Path("/dev/null/nope"))
    assert qrc.cached_rewrite("hyde", "p", "m", lambda: "value") == "value"


def test_entries_are_written_atomically_and_readably():
    qrc.cached_rewrite("hyde", "p", "m", lambda: "doc")
    entries = list(qrc.CACHE_DIR.glob("*.json"))
    assert len(entries) == 1
    payload = json.loads(entries[0].read_text(encoding="utf-8"))
    assert payload["kind"] == "hyde" and payload["result"] == "doc"
    assert not list(qrc.CACHE_DIR.glob("*.tmp"))


def test_clear_removes_entries():
    qrc.cached_rewrite("hyde", "p1", "m", lambda: "a")
    qrc.cached_rewrite("hyde", "p2", "m", lambda: "b")
    assert qrc.clear() == 2
    assert qrc.clear() == 0


# ----- the rewrite call sites ------------------------------------------------
# These exist because a NameError in decompose_query's transport shipped once:
# the cache tests all passed, since nothing exercised the call sites.


def test_decompose_query_goes_through_the_configured_provider(tmp_path, monkeypatch):
    from re_source import llm_utils

    monkeypatch.setattr(qrc, "CACHE_DIR", tmp_path / "c")

    calls = []

    def _generate(**kw):
        calls.append(kw)
        return json.dumps({"query": "Camp Lejeune contaminated water"})

    monkeypatch.setattr(llm_utils, "_llm_generate", _generate)

    query = "is there anything about contaminated water in Camp Lejeune"
    assert llm_utils.decompose_query(query) == ["Camp Lejeune contaminated water"]
    # Second call is served from cache, so the provider is hit once.
    assert llm_utils.decompose_query(query) == ["Camp Lejeune contaminated water"]
    assert len(calls) == 1
    assert calls[0]["json_mode"] is True


def test_decompose_query_falls_back_on_provider_failure(tmp_path, monkeypatch):
    from re_source import llm_utils

    monkeypatch.setattr(qrc, "CACHE_DIR", tmp_path / "c")

    calls = []

    def _generate(**_kw):
        calls.append(1)
        raise RuntimeError("boom")

    monkeypatch.setattr(llm_utils, "_llm_generate", _generate)

    query = "anything about contaminated water in Camp Lejeune"
    assert llm_utils.decompose_query(query) == [query]
    assert llm_utils.decompose_query(query) == [query]
    assert len(calls) == 2, "a transient failure must not be cached"


def test_hyde_and_expand_call_sites_run(tmp_path, monkeypatch):
    from re_source import llm_utils

    monkeypatch.setattr(qrc, "CACHE_DIR", tmp_path / "c")
    monkeypatch.setattr(llm_utils, "_llm_generate",
                        lambda **kw: json.dumps({"expanded_query": "teachers educators"})
                        if kw.get("json_mode") else "  a hypothetical tip  ")

    assert llm_utils.generate_hyde_document("teachers") == "a hypothetical tip"
    assert llm_utils.expand_query("teachers") == "teachers educators"


# ----- a rewrite has to be a string ------------------------------------------
#
# json_mode guarantees the response parses as an object, not that a given key
# holds a string. A non-string is truthy, so it survived every falsiness check
# on the path and reached _escape_embed_text as a dict — and was written to
# the cache first, which made that one query fail identically for every user
# from then on. The cache is documented as fail-open; this is the shape that
# got past it.

@pytest.mark.parametrize(
    "value",
    [{"terms": ["a"]}, ["a", "b"], 42, True, {"nested": {"deep": 1}}],
)
def test_a_non_string_expansion_falls_back_to_the_query(tmp_path, monkeypatch, value):
    from re_source import llm_utils

    monkeypatch.setattr(qrc, "CACHE_DIR", tmp_path / "c")
    monkeypatch.setattr(
        llm_utils, "_llm_generate",
        lambda **kw: json.dumps({"expanded_query": value}),
    )
    assert llm_utils.expand_query("teachers") == "teachers"


def test_a_blank_expansion_falls_back_to_the_query(tmp_path, monkeypatch):
    from re_source import llm_utils

    monkeypatch.setattr(qrc, "CACHE_DIR", tmp_path / "c")
    monkeypatch.setattr(
        llm_utils, "_llm_generate", lambda **kw: json.dumps({"expanded_query": "   "})
    )
    assert llm_utils.expand_query("teachers") == "teachers"


def test_the_fallback_is_what_gets_cached(tmp_path, monkeypatch):
    """The permanence is the whole problem: if the bad value were cached, the
    next call would reuse it without ever consulting the model again."""
    from re_source import llm_utils

    monkeypatch.setattr(qrc, "CACHE_DIR", tmp_path / "c")
    monkeypatch.setattr(
        llm_utils, "_llm_generate",
        lambda **kw: json.dumps({"expanded_query": {"terms": ["a"]}}),
    )
    llm_utils.expand_query("teachers")

    # Second call reads the cache; it must be the usable fallback, not the dict.
    monkeypatch.setattr(
        llm_utils, "_llm_generate",
        lambda **kw: (_ for _ in ()).throw(AssertionError("should be cached")),
    )
    assert llm_utils.expand_query("teachers") == "teachers"


# ----- a cached entry is only as trustworthy as what wrote it ----------------
#
# What fills this cache is a model response, and the caller's checks on that
# response live inside `compute` — which a hit skips entirely. So a validated
# call site does not protect a reader of an entry written before the check
# existed. Keyed on a hash of the prompt, one malformed answer was permanent
# for that exact query.

def _write_entry(tmp_path, kind, prompt, model_id, result):
    """Put a raw value in the cache the way _write would."""
    monkey_dir = tmp_path / "c"
    monkey_dir.mkdir(parents=True, exist_ok=True)
    key = qrc._key(kind, prompt, model_id)
    (monkey_dir / f"{key}.json").write_text(json.dumps({
        "kind": kind, "model_id": model_id, "result": result,
    }))
    return key


def test_a_malformed_cached_entry_is_not_returned(tmp_path, monkeypatch):
    monkeypatch.setattr(qrc, "CACHE_DIR", tmp_path / "c")
    _write_entry(tmp_path, "expand_query", "p", "m", {"terms": ["a"]})

    out = qrc.cached_rewrite(
        "expand_query", "p", "m", lambda: "recomputed",
        valid=lambda v: isinstance(v, str),
    )
    assert out == "recomputed"


def test_a_malformed_cached_entry_is_discarded_from_disk(tmp_path, monkeypatch):
    """Otherwise every later call pays to recompute and rediscover it."""
    monkeypatch.setattr(qrc, "CACHE_DIR", tmp_path / "c")
    _write_entry(tmp_path, "expand_query", "p", "m", {"terms": ["a"]})

    qrc.cached_rewrite(
        "expand_query", "p", "m", lambda: "recomputed",
        valid=lambda v: isinstance(v, str),
    )
    assert qrc.cached_rewrite(
        "expand_query", "p", "m",
        lambda: (_ for _ in ()).throw(AssertionError("should be cached now")),
        valid=lambda v: isinstance(v, str),
    ) == "recomputed"


def test_a_malformed_computed_value_is_not_stored(tmp_path, monkeypatch):
    monkeypatch.setattr(qrc, "CACHE_DIR", tmp_path / "c")

    qrc.cached_rewrite(
        "expand_query", "p", "m", lambda: {"terms": ["a"]},
        valid=lambda v: isinstance(v, str),
    )
    assert list((tmp_path / "c").glob("*.json")) == []


def test_a_good_entry_is_still_served_from_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(qrc, "CACHE_DIR", tmp_path / "c")
    _write_entry(tmp_path, "expand_query", "p", "m", "teachers educators")

    assert qrc.cached_rewrite(
        "expand_query", "p", "m",
        lambda: (_ for _ in ()).throw(AssertionError("should not recompute")),
        valid=lambda v: isinstance(v, str),
    ) == "teachers educators"


def test_no_validator_keeps_the_previous_behaviour(tmp_path, monkeypatch):
    monkeypatch.setattr(qrc, "CACHE_DIR", tmp_path / "c")
    _write_entry(tmp_path, "expand_query", "p", "m", {"terms": ["a"]})

    assert qrc.cached_rewrite("expand_query", "p", "m", lambda: "x") == {"terms": ["a"]}


def test_a_poisoned_expansion_on_disk_no_longer_reaches_the_caller(tmp_path, monkeypatch):
    """End to end, through the real call site — this is the hole a call-site
    isinstance check alone left open."""
    from re_source import llm_utils

    monkeypatch.setattr(qrc, "CACHE_DIR", tmp_path / "c")
    prompt_holder = {}

    def _capture(**kw):
        prompt_holder["p"] = kw["messages"][0]["content"]
        return json.dumps({"expanded_query": "teachers educators"})

    monkeypatch.setattr(llm_utils, "_llm_generate", _capture)
    llm_utils.expand_query("teachers")          # learn the real prompt + key

    _write_entry(tmp_path, "expand_query", prompt_holder["p"],
                 llm_utils._llm_identity(), {"terms": ["a"]})

    assert llm_utils.expand_query("teachers") == "teachers educators"


# ----- decompose reads a model value too -------------------------------------

@pytest.mark.parametrize("value", [{"a": 1}, ["a", "b"], 42, "", "   ", None])
def test_a_non_text_decomposition_falls_back_to_the_query(tmp_path, monkeypatch, value):
    from re_source import llm_utils

    monkeypatch.setattr(qrc, "CACHE_DIR", tmp_path / "c")
    monkeypatch.setattr(
        llm_utils, "_llm_generate", lambda **kw: json.dumps({"query": value})
    )

    assert llm_utils.decompose_query("tips about mold in housing") == [
        "tips about mold in housing"
    ]


def test_a_real_decomposition_still_comes_through(tmp_path, monkeypatch):
    from re_source import llm_utils

    monkeypatch.setattr(qrc, "CACHE_DIR", tmp_path / "c")
    monkeypatch.setattr(
        llm_utils, "_llm_generate",
        lambda **kw: json.dumps({"query": "Camp Lejeune contaminated water"}),
    )

    assert llm_utils.decompose_query("anything about Camp Lejeune water") == [
        "Camp Lejeune contaminated water"
    ]
