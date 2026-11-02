#!/usr/bin/env python3
"""Batch search: run multiple queries across all strategies, dedupe, LLM validate all."""

import argparse
import csv
import queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from re_source.search_runner import run_search
from re_source.vespa_strategies import VESPA_STRATEGIES
from re_source.llm_utils import validate_relevance_batch
from re_source.config import LLM_VALIDATION_BATCH_SIZE
from langfuse import observe, get_client


@observe(name="batch_search_query")
def run_all_strategies_for_query(query: str, strategies: dict, top_k: int = 100) -> list[dict]:
    """Run all strategies for a single query, return deduped results."""
    all_results = []

    print(f"\n[Query] {query}")
    for strategy_name in strategies.keys():
        t0 = time.time()
        try:
            result = run_search(strategy_name, query, top_k=top_k)
            for r in result.results:
                chunks = r.payload.get("matching_chunks", []) if r.payload else []
                all_results.append({
                    "doc_id": r.doc_id,
                    "source_file": r.source_file,
                    "score": r.score,
                    "matching_chunks": chunks,
                    "text": r.text,
                })
            print(f"  {strategy_name}: {len(result.results)} results ({time.time() - t0:.1f}s)")
        except Exception as e:
            print(f"  [ERROR] {strategy_name}: {e}")

    # Dedupe by doc_id, keeping best score
    seen = {}
    for r in all_results:
        doc_id = r["doc_id"]
        if doc_id not in seen or r["score"] > seen[doc_id]["score"]:
            seen[doc_id] = r

    deduped = list(seen.values())
    print(f"  Deduped: {len(all_results)} -> {len(deduped)}")
    return deduped


def load_existing_results(output_path: Path) -> set[str]:
    """The record ids already in a CSV, for --merge."""
    if not output_path.exists():
        return set()

    existing = set()
    with open(output_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ident = row.get("doc_id", "")
            if ident:
                existing.add(ident)
    return existing


@observe(name="batch_llm_validate")
def llm_validate_results(query: str, results: list[dict]) -> list[dict]:
    """Run LLM validation on results for a query."""
    if not results:
        return []

    print(f"  [LLM] Validating {len(results)} results...")

    # Build validation texts
    texts = []
    for r in results:
        if r["matching_chunks"]:
            chunk_texts = [c["text"] for c in r["matching_chunks"][:5]]
            texts.append("\n".join(chunk_texts))
        elif r["text"]:
            texts.append(r["text"])
        else:
            texts.append("")

    # Validate in batches
    all_validations = []
    for i in range(0, len(texts), LLM_VALIDATION_BATCH_SIZE):
        batch_texts = texts[i:i + LLM_VALIDATION_BATCH_SIZE]
        batch_num = i // LLM_VALIDATION_BATCH_SIZE + 1
        total_batches = (len(texts) + LLM_VALIDATION_BATCH_SIZE - 1) // LLM_VALIDATION_BATCH_SIZE

        t0 = time.time()
        validations = validate_relevance_batch(query, batch_texts)
        print(f"    Batch {batch_num}/{total_batches}: {time.time() - t0:.1f}s")
        all_validations.extend(validations)

    # Filter to validated only
    validated = []
    for r, v in zip(results, all_validations):
        if v.get("relevant", False):
            validated.append(r)

    print(f"  [LLM] Validated: {len(validated)}/{len(results)}")
    return validated


def export_query_results(results: list[dict], output_path: str):
    """Export results to CSV: the record, where it came from, and what matched."""
    rows = []
    for r in results:
        source = r["source_file"]

        # Format chunks
        chunks = r.get("matching_chunks", [])
        if chunks:
            # Format as "chunk 1", "chunk 2", etc.
            chunk_strs = []
            for c in chunks:
                text = c.get("text", "").replace('"', "'").replace('\n', ' ').strip()
                chunk_strs.append(f'"{text}"')
            chunks_formatted = ", ".join(chunk_strs)
        else:
            # Fall back to text preview
            text = (r.get("text") or "").replace('"', "'").replace('\n', ' ').strip()
            chunks_formatted = f'"{text[:500]}"' if text else ""

        rows.append({
            "doc_id": r.get("doc_id", ""),
            "source_file": source,
            "matching_chunks": chunks_formatted,
        })

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["doc_id", "source_file", "matching_chunks"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Exported {len(rows)} results to {output_path}")


def llm_worker(work_queue: queue.Queue, output_dir: Path, results_lock: threading.Lock, completed: list, merge: bool = False):
    """Worker thread that pulls from queue and does LLM validation."""
    while True:
        try:
            item = work_queue.get(timeout=1)
        except queue.Empty:
            continue

        if item is None:  # Poison pill - shutdown signal
            work_queue.task_done()
            break

        idx, output_name, query_text, results = item

        try:
            output_path = output_dir / f"{idx:02d}_{output_name}.csv"

            # If merging, filter out already-existing results
            if merge:
                existing = load_existing_results(output_path)
                original_count = len(results)
                results = [r for r in results if r.get("doc_id", "") not in existing]
                print(f"[LLM] {idx:02d}_{output_name}: {original_count} results, {len(results)} new (filtered {original_count - len(results)} existing)")

            if not results:
                print(f"[LLM] {idx:02d}_{output_name}: No new results to validate")
                with results_lock:
                    completed.append((f"{idx:02d}_{output_name}", 0))
                work_queue.task_done()
                continue

            print(f"[LLM] Starting: {idx:02d}_{output_name} ({len(results)} to validate)")
            validated = llm_validate_results(query_text, results)

            # If merging, append to existing file
            if merge and output_path.exists():
                # Read existing
                existing_rows = []
                with open(output_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    existing_rows = list(reader)

                # Add new validated results
                for r in validated:
                    chunks = r.get("matching_chunks", [])
                    if chunks:
                        chunk_strs = [f'"{c.get("text", "").replace(chr(34), chr(39)).replace(chr(10), " ").strip()}"' for c in chunks]
                        chunks_formatted = ", ".join(chunk_strs)
                    else:
                        text = (r.get("text") or "").replace('"', "'").replace('\n', ' ').strip()
                        chunks_formatted = f'"{text[:500]}"' if text else ""
                    existing_rows.append({
                        "source_file": r["source_file"],
                        "matching_chunks": chunks_formatted,
                    })

                # Write merged
                with open(output_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=["source_file", "matching_chunks"])
                    writer.writeheader()
                    writer.writerows(existing_rows)

                print(f"[LLM] Done: {idx:02d}_{output_name} (+{len(validated)} new, {len(existing_rows)} total)")
            else:
                export_query_results(validated, str(output_path))
                print(f"[LLM] Done: {idx:02d}_{output_name} ({len(validated)} validated)")

            with results_lock:
                completed.append((f"{idx:02d}_{output_name}", len(validated)))

        except Exception as e:
            import traceback
            print(f"[LLM] Error {idx:02d}_{output_name}: {e}")
            traceback.print_exc()
            with results_lock:
                completed.append((f"{idx:02d}_{output_name}", -1))

        work_queue.task_done()


# Strategies that do no LLM work during the search itself, so a sweep over them
# costs retrieval only. Names must exist in VESPA_STRATEGIES — this list used
# to carry two that had been renamed away, so --fast silently ran one strategy
# while its name implied three.
FAST_STRATEGIES = ["BM25"]


def parse_queries_file(path: str) -> list[tuple[str, str]]:
    """Read a queries file into [(output_name, query), ...]."""
    queries: list[tuple[str, str]] = []
    with open(path) as f:
        if path.endswith(".csv"):
            reader = csv.DictReader(f)
            for row in reader:
                # Expect columns: output_name, query (or filename, query)
                output_name = row.get("output_name") or row.get("filename") or row.get("name", "")
                query = row.get("query") or row.get("search", "")
                if output_name and query:
                    queries.append((output_name.strip(), query.strip()))
        else:
            # Text file: pipe-separated or just query
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "|" in line:
                    output_name, query = line.split("|", 1)
                    queries.append((output_name.strip(), query.strip()))
                else:
                    safe_name = line[:50].replace(" ", "_").replace("/", "_")
                    queries.append((safe_name, line))
    return queries


def main():
    parser = argparse.ArgumentParser(description="Batch search across all strategies")
    parser.add_argument("--file", "-f", required=True, help="File with queries (format: output_name|query)")
    parser.add_argument("--output-dir", "-o", default="batch_output", help="Output directory")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM validation")
    parser.add_argument("--llm-workers", type=int, default=3, help="Parallel LLM validation workers")
    parser.add_argument("--fast", action="store_true", help="Only run fast strategies (no LLM during search)")
    parser.add_argument("--strategies", "-s", nargs="+",
                        help=f"Only run these strategies (e.g. -s {' '.join(repr(k) for k in list(VESPA_STRATEGIES)[:2])})")
    parser.add_argument("--merge", action="store_true", help="Merge with existing results in output dir")
    args = parser.parse_args()

    # Filter strategies
    global strategies_to_run
    if args.strategies:
        strategies_to_run = {k: v for k, v in VESPA_STRATEGIES.items() if k in args.strategies}
        if not strategies_to_run:
            print(f"Error: No valid strategies. Available: {list(VESPA_STRATEGIES.keys())}")
            sys.exit(1)
    elif args.fast:
        strategies_to_run = {k: v for k, v in VESPA_STRATEGIES.items()
                             if k in FAST_STRATEGIES}
        if not strategies_to_run:
            print(f"Error: none of {FAST_STRATEGIES} is registered. "
                  f"Available: {list(VESPA_STRATEGIES)}")
            sys.exit(1)
    else:
        strategies_to_run = VESPA_STRATEGIES

    queries = parse_queries_file(args.file)

    print(f"Running {len(queries)} queries across {len(strategies_to_run)} strategies")
    print(f"Strategies: {list(strategies_to_run.keys())}")
    print(f"LLM workers: {args.llm_workers}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    t_start = time.time()
    # Vespa hosts the embedders server-side, so there's no local engine to load.

    if not args.no_llm:
        # Pipeline mode: search produces, LLM workers consume
        print("\n" + "=" * 60)
        print(f"PIPELINE MODE: Search -> LLM validation ({args.llm_workers} workers)")
        print("=" * 60)

        # Set up queue and workers
        work_queue = queue.Queue()
        results_lock = threading.Lock()
        completed = []

        # Start LLM worker threads
        workers = []
        for i in range(args.llm_workers):
            t = threading.Thread(target=llm_worker, args=(work_queue, output_dir, results_lock, completed, args.merge))
            t.start()
            workers.append(t)

        # Producer: run searches and feed queue
        for idx, (output_name, query_text) in enumerate(queries, start=1):
            results = run_all_strategies_for_query(query_text, strategies_to_run)
            if results:
                work_queue.put((idx, output_name, query_text, results))
                print(f"[Search] Queued: {idx:02d}_{output_name} ({len(results)} results)")
            else:
                print(f"[Search] No results: {idx:02d}_{output_name}")

        print(f"\n[Search] All searches complete, waiting for LLM workers...")

        # Send poison pills to stop workers
        for _ in workers:
            work_queue.put(None)

        # Wait for all work to complete
        for t in workers:
            t.join()

        print(f"\n{'='*60}")
        print(f"Completed: {len([c for c in completed if c[1] >= 0])}/{len(queries)} queries")
        for name, count in completed:
            if count >= 0:
                print(f"  {name}: {count} results")
            else:
                print(f"  {name}: ERROR")

    else:
        # No LLM - just run searches and export
        print("\n" + "=" * 60)
        print("Running searches (no LLM validation)")
        print("=" * 60)

        for idx, (output_name, query_text) in enumerate(queries, start=1):
            results = run_all_strategies_for_query(query_text, strategies_to_run)
            output_path = output_dir / f"{idx:02d}_{output_name}.csv"
            export_query_results(results, str(output_path))
            print(f"[Done] {idx:02d}_{output_name}: {len(results)} results")

    print(f"\n{'='*60}")
    print(f"Done! Total time: {time.time() - t_start:.1f}s")
    print(f"Results saved to: {output_dir}/")

    # Flush Langfuse traces before exiting
    try:
        get_client().flush()
    except Exception:
        pass


if __name__ == "__main__":
    main()
