"""Configuration and environment setup."""

import logging
import os
from pathlib import Path
from dotenv import load_dotenv
import yaml

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
# Overridable so the corpus does not have to sit inside the package tree:
# a deployment can point it at a mounted volume, and the private half of
# this repo keeps its own config and ground truth outside the public one.
DATA_DIR = Path(os.getenv("DATA_DIR", str(PROJECT_ROOT / "data")))

# DuckDB paths (for incremental indexing)
DUCKDB_PATH = Path(os.getenv("DUCKDB_PATH", str(DATA_DIR / "submissions_data.duckdb")))
VESPA_SYNC_STATE_PATH = Path(os.getenv("VESPA_SYNC_STATE_PATH", str(DATA_DIR / "vespa_sync_state.json")))
# Tracks which rows' author_profile enrichment chunks have been fed to Vespa
# (doc_id -> source content_hash). Separate from VESPA_SYNC_STATE_PATH so the
# additive enrichment feed never disturbs the field-chunk sync state.
ENRICHMENT_SYNC_STATE_PATH = Path(os.getenv("ENRICHMENT_SYNC_STATE_PATH", str(DATA_DIR / "enrichment_sync_state.json")))

# Record ID column name in imported data
RECORD_ID_COLUMN = os.getenv("RECORD_ID_COLUMN", "id")
DOCUMENT_ID_SCHEME = os.getenv("DOCUMENT_ID_SCHEME", "dataset_scoped").strip().lower()
if DOCUMENT_ID_SCHEME not in {"dataset_scoped", "legacy"}:
    raise ValueError("DOCUMENT_ID_SCHEME must be 'dataset_scoped' or 'legacy'")

# Search configuration
MIN_RESULTS = 0  # Allow returning zero results when nothing is relevant
COLBERT_MIN_SCORE = 0.45  # Absolute threshold on normalized ColBERT score (0-1 range)

# Evaluation configuration
EVAL_DATA_DIR = DATA_DIR / "eval"
# Batch size for the offline sweep in scripts/batch_search.py. The live
# strategies do NOT read this — each declares its own VALIDATION_BATCH_SIZE
# (12), sized for the model they actually call.
LLM_VALIDATION_BATCH_SIZE = 4
LLM_VALIDATION_MAX_RESULTS = 200  # Generous cap on results to validate with LLM;
# high enough that retrieval, not the cap, decides recall (recall-first goal),
# while still bounding pathological blow-ups in cost/latency.
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:14b")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "local")
LLM_PROVIDER_MODULE = os.getenv("LLM_PROVIDER_MODULE", "")
CHUNK_POLICY_MODULE = os.getenv("CHUNK_POLICY_MODULE", "")
EXPORT_POLICY_MODULE = os.getenv("EXPORT_POLICY_MODULE", "")
FILTER_POLICY_MODULE = os.getenv("FILTER_POLICY_MODULE", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_HOSTED_MODEL = os.getenv("LLM_HOSTED_MODEL", "gpt-4.1")
DEFAULT_K_VALUES = [5, 10, 20]

# Evaluation configuration paths
EVAL_CONFIG_PATH = DATA_DIR / "eval_config.yaml"
GROUND_TRUTH_DIR = DATA_DIR / "ground_truth"


def load_eval_config() -> dict:
    """Load evaluation configuration from YAML file.

    Returns:
        Dict with eval config, or defaults if file doesn't exist.
    """
    defaults = {
        "exclude_sources": [],
        "default_top_k": 5000,
    }

    if not EVAL_CONFIG_PATH.exists():
        return defaults

    try:
        with open(EVAL_CONFIG_PATH) as f:
            config = yaml.safe_load(f) or {}
        return {**defaults, **config}
    except Exception as e:
        logger.warning(f"Failed to load eval config: {e}")
        return defaults


def load_ground_truth_index() -> dict:
    """Load ground truth index mapping queries to CSV files.

    Returns:
        Dict with 'queries' list containing query definitions.
    """
    index_path = GROUND_TRUTH_DIR / "index.yaml"
    if not index_path.exists():
        return {"queries": []}

    try:
        with open(index_path) as f:
            return yaml.safe_load(f) or {"queries": []}
    except Exception as e:
        logger.warning(f"Failed to load ground truth index: {e}")
        return {"queries": []}


# Audit trail. Records which user saw or exported which records — a different
# concern from the Langfuse tracing above, which is LLM observability and is
# flushed best-effort. Audit events need durability and long retention, so they
# get their own append-only sink.
#   "s3"   -> one JSONL object per event in AUDIT_S3_BUCKET (deployed default)
#   "file" -> append to AUDIT_SPOOL_PATH (local dev; no AWS credentials needed)
#   "off"  -> record nothing
AUDIT_SINK = os.getenv("AUDIT_SINK", "file").lower()
AUDIT_S3_BUCKET = os.getenv("AUDIT_S3_BUCKET", "")
AUDIT_S3_PREFIX = os.getenv("AUDIT_S3_PREFIX", "audit").strip("/")
# Falls back to the ambient AWS region so a deployed task role needs no extra config.
AUDIT_S3_REGION = os.getenv("AUDIT_S3_REGION") or os.getenv("AWS_REGION") or None
# Also the landing spot for events the S3 sink failed to write, so a transient
# S3 outage degrades to a local file rather than dropping the record.
AUDIT_SPOOL_PATH = Path(os.getenv("AUDIT_SPOOL_PATH", str(DATA_DIR / "audit_spool.jsonl")))

# The longest search query either door accepts. Generous next to any real
# search — the longest sensible one is a sentence or two — but a bound at all,
# which neither door had. The query is re-sent in every LLM validation batch,
# so a caller-sized string is multiplied by the batch count before it reaches
# the provider, and it is written verbatim into the audit event.
MAX_QUERY_CHARS = 2000

# When true (default), the Streamlit app requires an authenticated user
# (st.login) before searching. Requires an OIDC provider configured in
# .streamlit/secrets.toml; set REQUIRE_LOGIN=false to allow anonymous use.
#
# Fails closed, matching api/auth.REQUIRE_API_AUTH: only an exact "false"
# turns the gate off, so REQUIRE_LOGIN=0, =no, or a typo leaves it on. This
# was `== "true"`, which fails the other way — every truthy spelling but
# "true" itself, and a value with a stray trailing space, silently disabled
# the only gate on the Streamlit surface and served the tip corpus to anyone
# who could reach the port, with nothing logged.
REQUIRE_LOGIN = os.getenv("REQUIRE_LOGIN", "true").strip().lower() != "false"
ADMIN_EMAILS = frozenset(
    value.strip().casefold()
    for value in os.getenv("ADMIN_EMAILS", "").split(",")
    if value.strip()
)
ADMIN_GROUPS = frozenset(
    value.strip().casefold()
    for value in os.getenv("ADMIN_GROUPS", "").split(",")
    if value.strip()
)
