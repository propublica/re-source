"""FastAPI app exposing Vespa search."""

from __future__ import annotations

import dataclasses

from fastapi import Depends, FastAPI, HTTPException
from starlette.concurrency import run_in_threadpool

from api.auth import Principal, require_search_user
from api.schemas import SearchRequest, SearchResponse, SearchResultOut
from re_source import audit
from re_source.search_runner import run_search
from re_source.vespa_strategies import VESPA_STRATEGIES

app = FastAPI(title="tips-search API", version="1.0.0")


@app.get("/health")
def health() -> dict:
    """Unauthenticated liveness probe. Reveals nothing about the corpus."""
    return {"status": "ok"}


@app.get("/strategies")
def strategies(user: Principal = Depends(require_search_user)) -> dict:
    return {"strategies": list(VESPA_STRATEGIES)}


@app.post("/search", response_model=SearchResponse)
async def search(
    req: SearchRequest, user: Principal = Depends(require_search_user)
) -> SearchResponse:
    if req.strategy not in VESPA_STRATEGIES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown strategy: {req.strategy!r}. "
            f"Available: {', '.join(VESPA_STRATEGIES)}",
        )
    # run_search is blocking (Vespa HTTP + LLM validation); offload so the event
    # loop stays responsive.
    result = await run_in_threadpool(run_search, req.strategy, req.query, req.top_k)
    # Records leaving through the API are the same disclosure as records leaving
    # through the UI, just a different door — so they land in the same audit log.
    # No session_id: a token-bearing caller has no equivalent of a browser session.
    audit.log_event(
        audit.EVENT_RESULTS_VIEWED,
        actor=user.user_id,
        record_ids=audit.record_refs(result.results),
        surface=audit.SURFACE_API,
        query=req.query,
        strategy=req.strategy,
    )
    return SearchResponse(
        approach_name=result.approach_name,
        elapsed_seconds=result.elapsed_seconds,
        metadata=result.metadata or {},
        results=[SearchResultOut(**dataclasses.asdict(r)) for r in result.results],
    )
