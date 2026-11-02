"""The LLM registry dispatches explicitly and fails closed."""

import pytest

from re_source import config
from re_source import llm_providers as providers
from re_source import llm_utils


def test_builtin_identities_follow_current_configuration(monkeypatch):
    monkeypatch.setattr(config, "LLM_MODEL", "local-model")
    monkeypatch.setattr(config, "LLM_HOSTED_MODEL", "hosted-model")

    assert providers.get_provider("local").identity() == "local:local-model"
    assert providers.get_provider("hosted").identity() == "hosted:hosted-model"


def test_unknown_provider_fails_closed(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "locla")

    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER 'locla'"):
        llm_utils._llm_identity()


def test_registered_adapter_drives_identity_and_generation(monkeypatch):
    name = "test-adapter"
    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        return "answer"

    providers.register_provider(
        name,
        providers.LLMProvider(identity=lambda: "test:model", generate=generate),
    )
    monkeypatch.setattr(config, "LLM_PROVIDER", name)

    assert llm_utils._llm_identity() == "test:model"
    assert llm_utils._llm_generate(
        name="test", messages=[{"role": "user", "content": "hello"}]
    ) == "answer"
    assert calls == [{
        "name": "test",
        "messages": [{"role": "user", "content": "hello"}],
        "json_mode": False,
    }]


def test_duplicate_registration_is_rejected():
    provider = providers.LLMProvider(identity=lambda: "x", generate=lambda **_: "x")
    with pytest.raises(ValueError, match="already registered"):
        providers.register_provider("local", provider)


class _Response:
    def raise_for_status(self):
        return None

    def json(self):
        return {"choices": [{"message": {"content": "hosted answer"}}]}


def test_hosted_adapter_posts_chat_completions(monkeypatch):
    sent = {}

    def post(url, **kwargs):
        sent.update(url=url, **kwargs)
        return _Response()

    monkeypatch.setattr(config, "LLM_BASE_URL", "https://llm.example/v1/")
    monkeypatch.setattr(config, "LLM_API_KEY", "secret")
    monkeypatch.setattr(config, "LLM_HOSTED_MODEL", "model-a")
    monkeypatch.setattr(providers.httpx, "post", post)

    result = providers.get_provider("hosted").generate(
        name="test",
        messages=[{"role": "user", "content": "hello"}],
        json_mode=True,
    )

    assert result == "hosted answer"
    assert sent["url"] == "https://llm.example/v1/chat/completions"
    assert sent["headers"] == {"Authorization": "Bearer secret"}
    assert sent["json"]["model"] == "model-a"
    assert sent["json"]["response_format"] == {"type": "json_object"}


def test_hosted_adapter_requires_a_base_url(monkeypatch):
    monkeypatch.setattr(config, "LLM_BASE_URL", "")
    with pytest.raises(ValueError, match="LLM_BASE_URL"):
        providers.get_provider("hosted").generate(
            name="test",
            messages=[],
        )
