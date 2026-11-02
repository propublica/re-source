"""The model fetch downloads e5 from a pinned, checksummed Hugging Face revision."""

import re
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fetch_vespa_models.sh"
SERVICES = Path(__file__).resolve().parents[1] / "vespa" / "services.xml"


def test_e5_is_downloaded_from_a_pinned_revision():
    script = SCRIPT.read_text()
    # A floating `main` would let a republished export change what a fresh
    # install embeds, silently, relative to an existing index.
    assert re.search(r'E5_REVISION="[0-9a-f]{40}"', script)
    assert "https://huggingface.co/intfloat/e5-large-v2/resolve/$E5_REVISION" in script


def test_e5_files_are_checksummed():
    script = re.sub(r"\\\n\s*", " ", SCRIPT.read_text())  # join continued lines
    assert re.search(r'"\$E5_BASE/onnx/model\.onnx"\s+\S+\s+[0-9a-f]{64}', script)
    assert re.search(r'"\$E5_BASE/tokenizer\.json"\s+\S+\s+[0-9a-f]{64}', script)


def test_e5_is_no_longer_exported_locally():
    assert "optimum" not in SCRIPT.read_text()


def test_e5_embedder_matches_feature_extraction_output():
    services = SERVICES.read_text()
    assert "<transformer-output>last_hidden_state</transformer-output>" in services
    assert "<pooling-strategy>mean</pooling-strategy>" in services


def test_a_cache_miss_does_not_end_the_script(tmp_path):
    """hf_cached is assigned under `set -e`; a miss returning non-zero used to
    stop the fetch silently before any download."""
    import subprocess

    text = SCRIPT.read_text()
    start = text.index("hf_cached() {")
    function = text[start:text.index("\n}\n", start) + 3]
    result = subprocess.run(
        ["bash", "-c", f'set -euo pipefail\n{function}\n'
                       'cached=$(hf_cached models--none--none model.onnx)\n'
                       'echo "continued:$cached"'],
        env={"HF_CACHE": str(tmp_path), "PATH": "/usr/bin:/bin"},
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "continued:"


def test_services_points_at_the_fetched_e5_revision():
    """Vespa caches a model by URL and never re-checks it, so the path must
    change whenever the model does: it carries the pinned revision."""
    revision = re.search(r'E5_REVISION="([0-9a-f]{40})"', SCRIPT.read_text()).group(1)
    services = SERVICES.read_text()
    assert f'url="http://models/e5-{revision[:7]}/model.onnx"' in services
    assert f'url="http://models/e5-{revision[:7]}/tokenizer.json"' in services
    assert 'url="http://models/e5/' not in services
