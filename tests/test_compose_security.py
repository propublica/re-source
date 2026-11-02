from pathlib import Path


PUBLIC_ROOT = Path(__file__).resolve().parents[1]


def test_public_service_ports_are_loopback_only():
    compose = (PUBLIC_ROOT / "docker-compose.yml").read_text()
    assert '"127.0.0.1:${VESPA_HTTP_PORT:-8080}:8080"' in compose
    assert '"127.0.0.1:${VESPA_CONFIG_PORT:-19071}:19071"' in compose
    assert '"127.0.0.1:${APP_UI_PORT:-8501}:8501"' in compose
    assert '"127.0.0.1:${APP_API_PORT:-8000}:8000"' in compose


def test_public_compose_uses_the_langfuse_sdk_switch():
    compose = (PUBLIC_ROOT / "docker-compose.yml").read_text()
    assert "LANGFUSE_TRACING_ENABLED" in compose


def test_public_models_are_mounted_from_the_public_tree():
    compose = (PUBLIC_ROOT / "docker-compose.yml").read_text()
    assert "./data/vespa_models:/usr/share/nginx/html:ro" in compose
    assert "../data/vespa_models" not in compose
