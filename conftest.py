"""Make the project root importable under pytest so tests can `import re_source.*`,
`import app`, and `import api.*` the same way the app does."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))


@pytest.fixture(autouse=True)
def _audit_off_by_default(monkeypatch):
    """Tests do not inherit private runtime choices from the developer shell."""
    from re_source import config

    monkeypatch.setattr(config, "AUDIT_SINK", "off")
    monkeypatch.setattr(config, "LLM_PROVIDER", "local")
    monkeypatch.setattr(config, "LLM_PROVIDER_MODULE", "")
    monkeypatch.setattr(config, "CHUNK_POLICY_MODULE", "")
    monkeypatch.setattr(config, "EXPORT_POLICY_MODULE", "")
    monkeypatch.setattr(config, "FILTER_POLICY_MODULE", "")


@pytest.fixture(autouse=True)
def _no_real_author_profiles(monkeypatch):
    """Feeds never read the developer's real enrichment table; tests opt in."""
    import re_source.embed_vespa as ev

    monkeypatch.setattr(ev, "_profiles_for", lambda docs: {})
