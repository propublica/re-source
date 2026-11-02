"""The lexical leg matches exhaustively rather than through weakAnd pruning."""

from re_source.vespa_strategies import (
    AtomicVespaStrategy,
    BM25VespaStrategy,
    _KEYWORD_CLAUSE,
    _LEXICAL_CLAUSE,
)


class _CapturingClient:
    """Records the YQL it was asked to run; returns no hits."""

    def __init__(self):
        self.yql = None
        self.last_kwargs = {}

    def query(self, **kwargs):
        self.yql = kwargs.get("yql")
        self.last_kwargs = kwargs
        return {"root": {"children": []}}

    def get(self, _schema, _doc_id):
        return {}


def _yql_for(strategy_cls, **kwargs):
    client = _CapturingClient()
    strategy_cls(client).execute("people wrongly removed from the voting rolls", **kwargs)
    return client.yql


def test_lexical_clause_is_exhaustive():
    assert _LEXICAL_CLAUSE == '({grammar: "any"}userInput(@query))'


def test_atomic_uses_the_exhaustive_lexical_leg():
    assert _LEXICAL_CLAUSE in _yql_for(AtomicVespaStrategy, top_k=500)


def test_keyword_strategy_uses_complete_parent_rows():
    yql = _yql_for(BM25VespaStrategy, top_k=500)
    assert "from parent" in yql
    assert _KEYWORD_CLAUSE in yql


def test_ann_leg_keeps_its_own_target_hits():
    yql = _yql_for(AtomicVespaStrategy, top_k=500)
    assert "{targetHits: 500}nearestNeighbor(dense, q_dense)" in yql


def test_keyword_strategy_does_not_apply_chunk_filters():
    yql = _yql_for(BM25VespaStrategy, top_k=100)
    assert "author_profile" not in yql
    assert "from chunk" not in yql


def test_no_unannotated_userinput_remains():
    atomic = _yql_for(AtomicVespaStrategy, top_k=500)
    keyword = _yql_for(BM25VespaStrategy, top_k=500)
    assert atomic.count("userInput(@query)") == atomic.count(_LEXICAL_CLAUSE)
    assert keyword.count("userInput(@query)") == keyword.count(_KEYWORD_CLAUSE)


def test_lexical_leg_does_not_depend_on_top_k():
    assert _yql_for(BM25VespaStrategy, top_k=10) == _yql_for(BM25VespaStrategy, top_k=500)


# ----- rank profile selection ------------------------------------------------


def test_atomic_uses_the_configured_profile(monkeypatch):
    import importlib
    import re_source.vespa_strategies as vs

    monkeypatch.setenv("VESPA_HYBRID_PROFILE", "hybrid_sat")
    reloaded = importlib.reload(vs)
    try:
        assert reloaded.HYBRID_PROFILE == "hybrid_sat"

        class _C:
            ranking = None

            def query(self, **kw):
                self.ranking = kw.get("ranking")
                return {"root": {"children": []}}

            def get(self, *_):
                return {}

        client = _C()
        reloaded.AtomicVespaStrategy(client).execute("q", top_k=10)
        assert client.ranking == "hybrid_sat"
    finally:
        monkeypatch.delenv("VESPA_HYBRID_PROFILE", raising=False)
        importlib.reload(vs)


# ----- rank profile and the bm25_k knob -------------------------------------


def _reload_with(monkeypatch, **env):
    import importlib
    import re_source.vespa_strategies as vs
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return importlib.reload(vs)


def _restore():
    import importlib
    import re_source.vespa_strategies as vs
    importlib.reload(vs)


class _InputCapturingClient:
    def __init__(self):
        self.kwargs = None

    def query(self, **kwargs):
        self.kwargs = kwargs
        return {"root": {"children": []}}

    def get(self, *_):
        return {}


def test_hybrid_sat_is_the_default_profile():
    from re_source.vespa_strategies import HYBRID_PROFILE
    assert HYBRID_PROFILE == "hybrid_sat"


def test_bm25_k_is_not_sent_unless_overridden():
    client = _InputCapturingClient()
    AtomicVespaStrategy(client).execute("q", top_k=10)
    assert "bm25_k" not in client.kwargs
    assert client.kwargs["ranking"] == "hybrid_sat"


def test_bm25_k_is_sweepable_without_redeploy(monkeypatch):
    vs = _reload_with(monkeypatch, VESPA_BM25_K="10")
    try:
        client = _InputCapturingClient()
        vs.AtomicVespaStrategy(client).execute("q", top_k=10)
        assert client.kwargs["bm25_k"] == "10"
    finally:
        monkeypatch.delenv("VESPA_BM25_K", raising=False)
        _restore()


def test_bm25_k_is_withheld_from_profiles_that_dont_declare_it(monkeypatch):
    vs = _reload_with(monkeypatch, VESPA_HYBRID_PROFILE="hybrid", VESPA_BM25_K="10")
    try:
        client = _InputCapturingClient()
        vs.AtomicVespaStrategy(client).execute("q", top_k=10)
        assert "bm25_k" not in client.kwargs
        assert client.kwargs["ranking"] == "hybrid"
    finally:
        monkeypatch.delenv("VESPA_HYBRID_PROFILE", raising=False)
        monkeypatch.delenv("VESPA_BM25_K", raising=False)
        _restore()


def test_stopword_stripping_can_be_disabled(monkeypatch):
    vs = _reload_with(monkeypatch, VESPA_LEXICAL_STOPWORDS="0")
    try:
        q = "anyone who has been a patient of"
        assert vs._lexical_text(q) == q
        client = _CapturingClient()
        vs.BM25VespaStrategy(client).execute(q, top_k=10)
        assert client.last_kwargs["query"] == q
    finally:
        monkeypatch.delenv("VESPA_LEXICAL_STOPWORDS", raising=False)
        _restore()


def test_stopword_stripping_is_on_by_default():
    from re_source.vespa_strategies import STRIP_LEXICAL_STOPWORDS
    assert STRIP_LEXICAL_STOPWORDS is True
