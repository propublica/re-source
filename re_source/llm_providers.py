"""LLM adapters behind a small registry shared by every calling workflow."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Callable

import httpx
from langfuse import observe

from . import config

Generate = Callable[..., str]


@dataclass(frozen=True)
class LLMProvider:
    identity: Callable[[], str]
    generate: Generate


_PROVIDERS: dict[str, LLMProvider] = {}
_loaded_module: str | None = None


def register_provider(name: str, provider: LLMProvider) -> None:
    """Register one adapter; duplicate names are configuration errors."""
    key = name.strip().lower()
    if not key:
        raise ValueError("LLM provider names cannot be empty")
    if key in _PROVIDERS:
        raise ValueError(f"LLM provider {key!r} is already registered")
    _PROVIDERS[key] = provider


def _load_configured_module() -> None:
    global _loaded_module
    module = config.LLM_PROVIDER_MODULE.strip()
    if module and module != _loaded_module:
        importlib.import_module(module)
        _loaded_module = module


def get_provider(name: str) -> LLMProvider:
    """Return a configured adapter, failing closed for unknown names."""
    _load_configured_module()
    key = name.strip().lower()
    try:
        return _PROVIDERS[key]
    except KeyError as exc:
        available = ", ".join(sorted(_PROVIDERS)) or "none"
        raise ValueError(
            f"Unknown LLM_PROVIDER {name!r}; registered providers: {available}"
        ) from exc


@observe(as_type="generation")
def _local_generation(
    *, name: str, messages: list[dict], json_mode: bool = False
) -> str:
    import ollama

    kwargs = {
        "model": config.LLM_MODEL,
        "messages": messages,
        "options": {"temperature": 0},
        "keep_alive": "1h",
    }
    if json_mode:
        kwargs["format"] = "json"
    response = ollama.chat(**kwargs)
    return response["message"]["content"]


@observe(as_type="generation")
def _hosted_generation(
    *, name: str, messages: list[dict], json_mode: bool = False
) -> str:
    base_url = config.LLM_BASE_URL.rstrip("/")
    if not base_url:
        raise ValueError("LLM_BASE_URL is required when LLM_PROVIDER='hosted'")

    body: dict = {
        "model": config.LLM_HOSTED_MODEL,
        "messages": messages,
        "temperature": 0,
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}

    headers = {}
    if config.LLM_API_KEY:
        headers["Authorization"] = f"Bearer {config.LLM_API_KEY}"
    response = httpx.post(
        f"{base_url}/chat/completions",
        headers=headers,
        json=body,
        timeout=60,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    if not isinstance(content, str):
        raise ValueError("Hosted LLM response did not contain text content")
    return content


register_provider(
    "local",
    LLMProvider(
        identity=lambda: f"local:{config.LLM_MODEL}",
        generate=_local_generation,
    ),
)
register_provider(
    "hosted",
    LLMProvider(
        identity=lambda: f"hosted:{config.LLM_HOSTED_MODEL}",
        generate=_hosted_generation,
    ),
)
