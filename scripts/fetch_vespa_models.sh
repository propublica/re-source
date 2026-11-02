#!/usr/bin/env bash
# Fetch e5-large-v2 (dense) + ColBERTv2 ONNX models into data/vespa_models/.
#
# Models live on the host at ./data/vespa_models/ and are served to Vespa
# by the `models` nginx sidecar (see docker-compose.vespa.yml). Vespa
# downloads them once via HTTP at first deploy, then caches in /opt/vespa/var.
#
# Files produced:
#   data/vespa_models/e5-<revision>/model.onnx      (~1.3 GB)
#   data/vespa_models/e5-<revision>/tokenizer.json
#   data/vespa_models/colbert/model.onnx      (~400 MB)
#   data/vespa_models/colbert/tokenizer.json
#
# Idempotent: skips files that are already complete. Re-run after
# `rm -rf data/vespa_models/<name>` to force re-fetch.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODELS_DIR="$REPO_ROOT/data/vespa_models"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}/hub"

hf_cached() {
  local repo_dir="$1" relative_path="$2"
  local match
  match=$(ls -1 "$HF_CACHE/$repo_dir"/snapshots/*/"$relative_path" 2>/dev/null | head -1)
  # A miss must still succeed: callers assign this under `set -e`, where a
  # non-zero status ends the script without a word.
  if [[ -n "$match" && -f "$match" ]]; then
    echo "$match"
  fi
}

fetch_curl() {
  local url="$1"
  local dest="$2"
  local hf_repo="$3"
  local hf_path="$4"
  mkdir -p "$(dirname "$dest")"

  local expected
  expected=$(curl -sIL "$url" | awk 'tolower($1)=="content-length:" {len=$2} END{gsub("\r","",len); print len}')

  if [[ -s "$dest" ]]; then
    local actual
    actual=$(wc -c < "$dest" | tr -d ' ')
    if [[ -n "$expected" && "$actual" == "$expected" ]]; then
      echo "  [skip] $dest is complete ($(du -h "$dest" | cut -f1))"
      return
    fi
  fi

  local cached
  cached=$(hf_cached "$hf_repo" "$hf_path")
  if [[ -n "$cached" ]]; then
    local cached_size
    cached_size=$(wc -c < "$cached" | tr -d ' ')
    if [[ -n "$expected" && "$cached_size" == "$expected" ]]; then
      echo "  [hflink] $cached -> $dest"
      rm -f "$dest"
      ln "$cached" "$dest" 2>/dev/null || cp "$cached" "$dest"
      return
    fi
  fi

  if [[ -s "$dest" ]]; then
    echo "  [resume] $dest is $(wc -c < "$dest" | tr -d ' ')/$expected bytes"
  else
    echo "  [get]  $url ($(numfmt --to=iec --suffix=B "$expected" 2>/dev/null || echo "$expected bytes"))"
  fi
  curl --fail --location --continue-at - --progress-bar --output "$dest" "$url"
}

sha256_of() {
  if command -v sha256sum >/dev/null; then
    sha256sum "$1" | cut -d' ' -f1
  else
    shasum -a 256 "$1" | cut -d' ' -f1
  fi
}

fetch_verified() {
  local url="$1" dest="$2" sha="$3" hf_repo="$4" hf_path="$5"
  mkdir -p "$(dirname "$dest")"

  if [[ -s "$dest" ]]; then
    if [[ "$(sha256_of "$dest")" != "$sha" ]]; then
      echo "  [fail] $dest does not match its checksum; delete it and re-run." >&2
      exit 1
    fi
    echo "  [skip] $dest is complete ($(du -h "$dest" | cut -f1))"
    return
  fi

  local cached
  cached=$(hf_cached "$hf_repo" "$hf_path")
  if [[ -n "$cached" && "$(sha256_of "$cached")" == "$sha" ]]; then
    echo "  [hflink] $cached -> $dest"
    ln "$cached" "$dest" 2>/dev/null || cp "$cached" "$dest"
    return
  fi

  # Resume into a side file, so an interrupted download never looks complete.
  echo "  [get]  $url"
  curl --fail --location --continue-at - --progress-bar --output "$dest.part" "$url"
  if [[ "$(sha256_of "$dest.part")" != "$sha" ]]; then
    echo "  [fail] $dest.part does not match its checksum; delete it and re-run." >&2
    exit 1
  fi
  mv "$dest.part" "$dest"
}

echo "==> e5-large-v2 (dense)"
# intfloat publishes an ONNX export of e5-large-v2. Pinned to one revision so a
# fresh install embeds exactly like an existing index; the checksums are that
# revision's files. The directory carries the revision because Vespa caches a
# model by URL: services.xml must name the same e5-<revision> path.
E5_REVISION="f169b11e22de13617baa190a028a32f3493550b6"
E5_BASE="https://huggingface.co/intfloat/e5-large-v2/resolve/$E5_REVISION"
E5_DIR="$MODELS_DIR/e5-${E5_REVISION:0:7}"
fetch_verified "$E5_BASE/onnx/model.onnx" "$E5_DIR/model.onnx" \
  f560bbc5da586764fc6ccf54990bac71f94e5a872ed4e1889039391c849cbcf8 \
  models--intfloat--e5-large-v2 onnx/model.onnx
fetch_verified "$E5_BASE/tokenizer.json" "$E5_DIR/tokenizer.json" \
  d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66 \
  models--intfloat--e5-large-v2 tokenizer.json
if [[ -d "$MODELS_DIR/e5" ]]; then
  echo "  [note] $MODELS_DIR/e5 is no longer used; delete it once Vespa serves ${E5_DIR##*/}."
fi

echo "==> ColBERTv2"
fetch_curl "https://huggingface.co/colbert-ir/colbertv2.0/resolve/main/model.onnx"     "$MODELS_DIR/colbert/model.onnx"     models--colbert-ir--colbertv2.0 model.onnx
fetch_curl "https://huggingface.co/colbert-ir/colbertv2.0/resolve/main/tokenizer.json" "$MODELS_DIR/colbert/tokenizer.json" models--colbert-ir--colbertv2.0 tokenizer.json

echo ""
echo "Done. Models in: $MODELS_DIR"
du -sh "$MODELS_DIR"/*/
