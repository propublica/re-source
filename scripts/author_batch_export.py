#!/usr/bin/env python3
"""Batch Author Profile search: several "who submitted this" queries, ONE export."""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.batch_search import parse_queries_file
from re_source.disclosure import default_actor, prepare_by_query_export, prepare_export
from re_source.search_runner import build_strategy

STRATEGY = "Author Profile + LLM"


def _evidence_key(evidence: dict) -> tuple[str, str]:
    return (evidence.get("field", ""), evidence.get("quote", ""))


def merge_results(per_query: list[tuple[str, list]]) -> list:
    """Union per-query results into one ranked list, de-duplicated by record.

    Args:
        per_query: [(query_label, results), ...] in the order they were run.
    """
    merged: dict[str, object] = {}
    for label, results in per_query:
        for r in results:
            kept = merged.get(r.doc_id)
            if kept is None:
                r.payload = dict(r.payload or {})
                r.payload["queries"] = [label]
                merged[r.doc_id] = r
                continue

            # Same record from a second query: fold in its label and evidence.
            payload = kept.payload
            if label not in payload["queries"]:
                payload["queries"].append(label)
            # A copy, so the first run's own payload list is not mutated.
            evidence = list(payload.get("evidence") or [])
            seen = {_evidence_key(e) for e in evidence}
            for item in (r.payload or {}).get("evidence") or []:
                if _evidence_key(item) not in seen:
                    seen.add(_evidence_key(item))
                    evidence.append(item)
            payload["evidence"] = evidence

            if r.score > kept.score:
                # Promote the better-scoring run's score and its reasoning, but
                # keep the accumulated queries/evidence built up above.
                kept.score = r.score
                if (r.payload or {}).get("llm_reason") is not None:
                    payload["llm_reason"] = r.payload["llm_reason"]

    return sorted(merged.values(), key=lambda r: r.score, reverse=True)


def snapshot(per_query: list[tuple[str, list]]) -> list[tuple[str, list]]:
    """Copies of each query's results, so merging can't swap in another query's reason."""
    # merge_results rewrites the first-seen result's payload and score with the
    # best-scoring query's; a per-query tab must show what THAT query decided.
    return [
        (label, [dataclasses.replace(r, payload=dict(r.payload or {})) for r in results])
        for label, results in per_query
    ]


def build_tuned_strategy(batch_size: int | None, workers: int | None):
    """The Author Profile strategy, with this run's LLM batching overrides."""
    strategy = build_strategy(STRATEGY)
    # Instance attributes shadow the class defaults for this run only; the UI's
    # strategies are built separately and keep theirs.
    if batch_size:
        strategy.VALIDATION_BATCH_SIZE = batch_size
    if workers:
        strategy.VALIDATION_WORKERS = workers
    return strategy


def run_queries(strategy, queries: list[tuple[str, str]], top_k: int,
                parallel: int) -> tuple[list[tuple[str, list]], list[str]]:
    """Run every query, `parallel` at a time.

    Returns:
        (per_query results in input order, labels of queries that failed).
    """
    done: dict[int, list] = {}
    failed: dict[int, str] = {}

    def _one(query: str):
        t0 = time.time()
        return strategy.execute(query=query, top_k=top_k), time.time() - t0

    with ThreadPoolExecutor(max_workers=max(1, parallel)) as ex:
        futures = {ex.submit(_one, query): (i, label)
                   for i, (label, query) in enumerate(queries)}
        for n, future in enumerate(as_completed(futures), start=1):
            i, label = futures[future]
            try:
                result, elapsed = future.result()
            except Exception as e:
                print(f"  [{n}/{len(queries)}] {label}: ERROR {e}", file=sys.stderr)
                failed[i] = label
                continue
            done[i] = result.results
            print(f"  [{n}/{len(queries)}] {label}: {len(result.results)} verified "
                  f"of {result.metadata.get('before_llm', '?')} ({elapsed:.1f}s)")

    # Input order, so the merge is the same however the threads finished.
    per_query = [(queries[i][0], done[i]) for i in sorted(done)]
    return per_query, [failed[i] for i in sorted(failed)]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch Author Profile search into one combined XLSX export."
    )
    parser.add_argument("--file", "-f", required=True,
                        help="Queries file: `label|query` per line, or a CSV with "
                             "output_name/query columns.")
    parser.add_argument("--out", "-o", default="author_batch_export.xlsx",
                        help="Output .xlsx path.")
    parser.add_argument("--top-k", type=int, default=100,
                        help="Candidates retrieved AND LLM-verified per query "
                             "(default 100; every candidate is verified, so this "
                             "is the cost knob).")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Tips per LLM validation call (default: the strategy's 12). "
                             "Smaller batches are more accurate on smaller models.")
    parser.add_argument("--workers", type=int, default=None,
                        help="Concurrent LLM calls per query (default: the strategy's 8).")
    parser.add_argument("--parallel-queries", type=int, default=1,
                        help="Queries run at once (default 1). In-flight LLM calls are "
                             "up to --parallel-queries x --workers; mind provider rate limits.")
    parser.add_argument("--tab-per-query", action="store_true",
                        help="After the Summary tab, one tab per query, each holding one "
                             "table per source dataset.")
    parser.add_argument("--actor", default=default_actor(),
                        help="Who this run is attributed to in the audit trail "
                             "(default: $AUDIT_ACTOR, else the OS user).")
    args = parser.parse_args()

    queries = parse_queries_file(args.file)
    if not queries:
        print(f"No queries found in {args.file}", file=sys.stderr)
        return 1

    print(f"{len(queries)} queries -> {STRATEGY}, top_k={args.top_k} each, "
          f"{args.parallel_queries} at a time")
    t_start = time.time()
    strategy = build_tuned_strategy(args.batch_size, args.workers)
    per_query, failed = run_queries(strategy, queries, args.top_k, args.parallel_queries)

    tabs = snapshot(per_query) if args.tab_per_query else None
    merged = merge_results(per_query)
    print(f"\nUnion: {sum(len(r) for _, r in per_query)} -> {len(merged)} unique records")
    if not merged:
        print("Nothing to export.")
        return 1 if failed else 0

    out_path = Path(args.out)
    # The workbook is the disclosure — writing it records one event covering
    # every record in it.
    if tabs is not None:
        disclosure = prepare_by_query_export(
            merged, tabs, actor=args.actor, strategy=STRATEGY,
        )
    else:
        disclosure = prepare_export(
            merged, actor=args.actor, strategy=STRATEGY,
        )
    disclosure.write(out_path)
    print(f"Wrote {len(merged)} records to {out_path} ({time.time() - t_start:.1f}s)")
    if failed:
        print(f"Queries that failed: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
