"""Keep the container on the Streamlit DOM that the shared CSS targets."""

from pathlib import Path
import tomllib

from re_source.brand import COLORS


PUBLIC_ROOT = Path(__file__).resolve().parents[1]


def test_streamlit_is_pinned_to_the_tested_ui_version():
    project = tomllib.loads((PUBLIC_ROOT / "pyproject.toml").read_text())
    assert "streamlit==1.58.0" in project["project"]["dependencies"]


def test_the_container_builds_from_the_public_lockfile():
    dockerfile = (PUBLIC_ROOT / "Dockerfile").read_text()
    assert "COPY pyproject.toml uv.lock ./" in dockerfile
    assert "uv sync --frozen --no-dev --no-install-project" in dockerfile


def test_streamlit_uses_the_brand_accent_instead_of_its_default_red():
    config = tomllib.loads(
        (PUBLIC_ROOT / ".streamlit" / "config.toml").read_text()
    )
    assert config["theme"]["primaryColor"] == COLORS["blue_03"]


def test_the_public_container_includes_the_streamlit_theme():
    dockerfile = (PUBLIC_ROOT / "Dockerfile").read_text()
    assert "COPY .streamlit/config.toml ./.streamlit/config.toml" in dockerfile
