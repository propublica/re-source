"""Thin wrapper around pyvespa for feed and query against the local Vespa
container defined in `docker-compose.vespa.yml`.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from typing import Any, Iterable, Iterator

from vespa.application import Vespa
from vespa.io import VespaResponse

logger = logging.getLogger(__name__)

VESPA_ENDPOINT = os.getenv("VESPA_ENDPOINT", "http://localhost:8080")

# Parallel slices for a full delete. pyvespa's default of 1 makes wiping a
# 700k-chunk index take hours; it fans slices out over a thread pool, so this
# maps straight onto concurrency.
DELETE_SLICES = int(os.getenv("VESPA_DELETE_SLICES", "8"))

# Concurrent feed connections. Both embedders run server-side, so each document
# costs two ONNX forward passes inside the container — feeding is bounded by how
# many of those are in flight, not by anything client-side. Measured at the
# default of 8: ~6 chunks/sec with the container at 15% CPU on 14 cores, i.e.
# twelve cores idle. Raise with VESPA_FEED_WORKERS and watch `docker stats`; if
# CPU stops climbing as workers go up, the container is the limit and the next
# lever is ONNX threading in services.xml.
FEED_WORKERS = int(os.getenv("VESPA_FEED_WORKERS", "8"))


@dataclass
class FeedResult:
    """Outcome of a batch feed."""
    successes: int
    failures: int
    failure_examples: list[dict]  # up to 5 sample failures for debugging


class VespaClient:
    """Feed and query against the tips-search Vespa application."""

    def __init__(self, endpoint: str = VESPA_ENDPOINT) -> None:
        self.endpoint = endpoint
        self.app = Vespa(url=endpoint)

    # ----- feed --------------------------------------------------------

    def feed_one(self, schema: str, doc_id: str, fields: dict[str, Any]) -> dict:
        """Feed a single document. Raises on HTTP error."""
        response = self.app.feed_data_point(
            schema=schema, data_id=doc_id, fields=fields
        )
        if not response.is_successful():
            raise RuntimeError(
                f"feed_one({schema}, {doc_id}) failed: {response.get_status_code()} {response.get_json()}"
            )
        return response.get_json()

    def feed_batch(
        self,
        schema: str,
        docs: Iterable[tuple[str, dict[str, Any]]],
        max_workers: int = FEED_WORKERS,
    ) -> FeedResult:
        """Feed many docs in parallel via pyvespa's feed_iterable.

        Args:
            schema: target schema name.
            docs: iterable of (doc_id, fields) tuples.
            max_workers: concurrent feed connections. Defaults to
                FEED_WORKERS (VESPA_FEED_WORKERS).

        Returns:
            FeedResult with success/failure counts.
        """
        successes = 0
        failure_count = 0
        failure_examples: list[dict] = []

        def _gen() -> Iterator[dict]:
            for doc_id, fields in docs:
                yield {"id": doc_id, "fields": fields}

        def _callback(response: VespaResponse, doc_id: str) -> None:
            nonlocal successes, failure_count
            if response.is_successful():
                successes += 1
            else:
                failure_count += 1
                if len(failure_examples) < 5:
                    failure_examples.append(
                        {"id": doc_id, "status": response.get_status_code(), "body": response.get_json()}
                    )

        self.app.feed_iterable(
            iter=_gen(),
            schema=schema,
            callback=_callback,
            max_workers=max_workers,
        )

        return FeedResult(
            successes=successes,
            failures=failure_count,
            failure_examples=failure_examples,
        )

    # ----- query -------------------------------------------------------

    def query(
        self,
        yql: str,
        ranking: str | None = None,
        hits: int = 10,
        timeout: str = "10s",
        **query_inputs: Any,
    ) -> dict:
        """Run a YQL query.

        Args:
            yql: YQL query string.
            ranking: rank profile name (e.g. "hybrid").
            hits: number of hits to return.
            timeout: Vespa query timeout (string like "10s").
            **query_inputs: passed as `input.query(name)=value` parameters —
                tensor inputs (q_dense, q_colbert) and `embed(...)` strings.
                The one exception is `query=`, which becomes Vespa's
                top-level user-query parameter, read by YQL
                `userInput(@query)`.

        Returns:
            The raw Vespa JSON response.
        """
        body: dict[str, Any] = {"yql": yql, "hits": hits, "timeout": timeout}
        if ranking:
            body["ranking.profile"] = ranking
        for name, value in query_inputs.items():
            # `query` is Vespa's top-level user-query parameter, read by
            # YQL `userInput(@query)`. Everything else is a rank-feature
            # input tensor or embed(...) call.
            if name == "query":
                body["query"] = value
            else:
                body[f"input.query({name})"] = value

        response = self.app.query(body=body)
        if not response.is_successful():
            raise RuntimeError(
                f"query failed: {response.get_status_code()} {response.get_json()}"
            )
        return response.get_json()

    # ----- get / delete ------------------------------------------------

    def get(self, schema: str, doc_id: str) -> dict | None:
        """Fetch one document by id. Returns the `fields` dict, or None if 404."""
        response = self.app.get_data(schema=schema, data_id=doc_id)
        if response.get_status_code() == 404:
            return None
        if not response.is_successful():
            raise RuntimeError(
                f"get({schema}, {doc_id}) failed: {response.get_status_code()} {response.get_json()}"
            )
        return response.get_json().get("fields")

    def delete(self, schema: str, doc_id: str) -> None:
        """Delete one document by id. Idempotent (no-op on 404)."""
        response = self.app.delete_data(schema=schema, data_id=doc_id)
        if response.get_status_code() in (200, 404):
            return
        raise RuntimeError(
            f"delete({schema}, {doc_id}) failed: {response.get_status_code()} {response.get_json()}"
        )

    def count(self, schema: str) -> int:
        """How many documents are in `schema`."""
        resp = self.query(
            yql=f"select * from {schema} where true", ranking="unranked", hits=0
        )
        return resp["root"]["fields"]["totalCount"]

    def delete_all_verified(
        self, schema: str, slices: int = DELETE_SLICES, attempts: int = 5
    ) -> int:
        """delete_all, then confirm the schema is actually empty, retrying while
        the count is still falling.
        """
        previous = None
        for attempt in range(1, attempts + 1):
            self.delete_all(schema, slices=slices)
            remaining = self.count(schema)
            if remaining == 0:
                logger.info(f"'{schema}' is empty")
                return 0
            logger.warning(
                f"'{schema}' still has {remaining:,} documents after delete "
                f"(attempt {attempt}/{attempts}) — retrying"
            )
            if previous is not None and remaining >= previous:
                raise RuntimeError(
                    f"delete_all('{schema}') stalled at {remaining:,} documents. "
                    "Feeding now would leave stale chunks from the previous "
                    "configuration in the index."
                )
            previous = remaining
        raise RuntimeError(
            f"delete_all('{schema}') left {previous:,} documents after "
            f"{attempts} attempts."
        )

    def delete_all(self, schema: str, slices: int = DELETE_SLICES) -> None:
        """Delete every document in `schema`."""
        logger.info(f"deleting all '{schema}' documents ({slices} parallel slices)")
        self.app.delete_all_docs(
            schema=schema, content_cluster_name="tips", slices=slices
        )

    def delete_by_selection(
        self,
        schema: str,
        selection: str,
        cluster: str = "tips",
        namespace: str | None = None,
    ) -> int:
        """Delete every document in `schema` matching a Vespa document selection
        expression.

        Returns:
            The number of HTTP rounds it took (visit may paginate).
        """
        import requests
        ns = namespace or schema
        base = f"{self.endpoint}/document/v1/{ns}/{schema}/docid/"
        params: dict[str, Any] = {"cluster": cluster, "selection": selection}
        rounds = 0
        while True:
            rounds += 1
            resp = requests.delete(base, params=params, timeout=120)
            resp.raise_for_status()
            body = resp.json()
            cont = body.get("continuation")
            if not cont:
                return rounds
            params["continuation"] = cont


# --- Shared client singleton -------------------------------------------------
# One VespaClient for the whole process, built lazily. Thread-safe so the
# Streamlit UI and the FastAPI threadpool (which may construct it concurrently
# on first use) don't race. VespaClient is a stateless HTTP wrapper, so a single
# shared instance is fine.
_client: VespaClient | None = None
_client_lock = threading.Lock()


def get_vespa_client() -> VespaClient:
    """Return the process-wide VespaClient, constructing it once on first call."""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = VespaClient()
    return _client
