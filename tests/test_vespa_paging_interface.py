"""Which strategies offer 'Load more', and what that commits them to."""

from re_source.vespa_strategies import (
    DEFAULT_PAGE_SIZE,
    VESPA_STRATEGIES,
    BM25VespaStrategy,
    Paged,
)


def _paginating():
    return {name for name, cls in VESPA_STRATEGIES.items()
            if getattr(cls, "paginates", False)}


def test_the_paginating_strategies_are_content_and_author_profile():
    assert _paginating() == {"Content", "Author Profile + LLM"}


def test_every_paginating_strategy_implements_the_pair():
    for name, cls in VESPA_STRATEGIES.items():
        if not getattr(cls, "paginates", False):
            continue
        assert callable(getattr(cls, "retrieve_pool", None)), name
        assert callable(getattr(cls, "validate_page", None)), name


def test_every_paginating_strategy_has_a_usable_page_size():
    for name, cls in VESPA_STRATEGIES.items():
        if not getattr(cls, "paginates", False):
            continue
        assert isinstance(cls.PAGE_SIZE, int), name
        assert cls.PAGE_SIZE > 0, name


def test_a_non_paginating_strategy_declares_nothing():
    assert getattr(BM25VespaStrategy, "paginates", False) is False
    assert not hasattr(BM25VespaStrategy, "PAGE_SIZE")


def test_the_page_size_has_one_definition():
    for cls in VESPA_STRATEGIES.values():
        if getattr(cls, "paginates", False):
            assert cls.PAGE_SIZE == DEFAULT_PAGE_SIZE


def test_the_mixin_carries_no_constructor():
    assert "__init__" not in vars(Paged)
