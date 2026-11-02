#!/usr/bin/env python3
"""Capture and compare retrieval snapshots, so changes to the stack are
falsifiable without a labelled eval set.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from re_source.vespa_client import get_vespa_client
from re_source.vespa_strategies import (
    DEFAULT_PAGE_SIZE,
    DEFAULT_TOP_K,
    HYBRID_PROFILE,
    AtomicVespaStrategy,
    BM25VespaStrategy,
    CombinedVespaStrategy,
)

SCHEMA_VERSION = 1


# Vespa's metrics proxy. Unverified against this deployment — if it isn't
# reachable the capture records null and carries on rather than failing.
VESPA_METRICS_URL = "http://localhost:19092/prometheus/v1/values"
MEMORY_METRIC_SUBSTRINGS = (
    "content.proton.documentdb.ready.attribute.memory_usage.allocated_bytes",
    "content.proton.documentdb.index.memory_usage.allocated_bytes",
    "content.proton.documentdb.documents.total",
)


# ----- capture ---------------------------------------------------------------


# Column names scripts/batch_search.py already accepts in a queries CSV.
_CSV_QUERY_COLUMNS = ("query", "search")


def load_queries(path: Path) -> list[str]:
    """Read a query set from .csv or a plain text file."""
    if path.suffix.lower() == ".csv":
        return _load_queries_csv(path)

    queries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        queries.append(line.split("|", 1)[1].strip() if "|" in line else line)
    return queries


def _load_queries_csv(path: Path) -> list[str]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return []
        lowered = {(name or "").strip().lower(): name for name in reader.fieldnames}
        column = next((lowered[c] for c in _CSV_QUERY_COLUMNS if c in lowered), None)
        if column is None:
            # No recognized header. `name,query` is the convention, so take the
            # last column rather than guessing the first.
            column = reader.fieldnames[-1]
            print(
                f"  [warn] {path.name} has no 'query' column; using "
                f"{column!r} (columns: {reader.fieldnames})",
                file=sys.stderr,
            )
        queries = []
        for row in reader:
            value = (row.get(column) or "").strip()
            if value and not value.startswith("#"):
                queries.append(value)
        return queries


def git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def vespa_memory() -> dict | None:
    """Snapshot of Vespa's attribute/index memory, for the item 8 sizing model.

    Returns:
        The memory snapshot, or None if the metrics endpoint isn't reachable.
    """
    try:
        import requests
        resp = requests.get(VESPA_METRICS_URL, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        print(f"  [warn] Vespa metrics unavailable ({e}); recording null", file=sys.stderr)
        return None

    out: dict[str, float] = {}
    for line in resp.text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        if not any(s in line for s in MEMORY_METRIC_SUBSTRINGS):
            continue
        name, _, value = line.rpartition(" ")
        try:
            out[name.strip()] = float(value)
        except ValueError:
            continue
    return out or None


def score_stats(scores: list[float]) -> dict | None:
    """min/median/max of a score list."""
    if not scores:
        return None
    return {
        "min": round(min(scores), 4),
        "median": round(statistics.median(scores), 4),
        "max": round(max(scores), 4),
        "count": len(scores),
    }


def ranked_entries(results, top_n: int) -> list[dict]:
    """Ranked parent ids + scores. No text, no full_row — see module docstring."""
    out = []
    for r in results[:top_n]:
        entry = {"parent_id": r.doc_id, "score": round(float(r.score), 4)}
        lanes = (r.payload or {}).get("lanes")
        if lanes:
            entry["lanes"] = list(lanes)
        out.append(entry)
    return out


def capture_query(client, query: str, top_k: int, top_n: int, repeat: int) -> dict:
    """Run the full Content pool plus the two deterministic strategies."""
    combined = CombinedVespaStrategy(client)
    atomic = AtomicVespaStrategy(client)
    bm25 = BM25VespaStrategy(client)

    pool_times: list[float] = []
    pool: list = []
    meta: dict = {}
    for _ in range(repeat):
        t0 = time.time()
        pool, meta = combined.retrieve_pool(query, top_k=top_k)
        pool_times.append(round(time.time() - t0, 3))

    deterministic = {}
    for name, strategy in (("atomic", atomic), ("bm25", bm25)):
        times: list[float] = []
        res = None
        for _ in range(repeat):
            t0 = time.time()
            res = strategy.execute(query=query, top_k=top_k)
            times.append(round(time.time() - t0, 3))
        deterministic[name] = {
            "ranked": ranked_entries(res.results, top_n),
            "elapsed_s": times,
            "score_stats": score_stats([float(r.score) for r in res.results]),
            "result_count": len(res.results),
        }

    return {
        "query": query,
        "pool": {
            "ranked": ranked_entries(pool, top_n),
            "meta": meta,
            "elapsed_s": pool_times,
            "score_stats": score_stats([float(r.score) for r in pool]),
            "result_count": len(pool),
        },
        "deterministic": deterministic,
    }


def quiet_http_logging(verbose: bool) -> None:
    """pyvespa logs every HTTP call at INFO through the `httpr` logger, which buries the
    capture's own output under hundreds of lines.
    """
    if verbose:
        return
    for name in ("httpr", "httpx", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)


def cmd_capture(args) -> int:
    quiet_http_logging(args.verbose)
    queries = load_queries(Path(args.queries))
    if not queries:
        print(f"No queries found in {args.queries}", file=sys.stderr)
        return 1

    if args.dry_run:
        print(f"Parsed {len(queries)} queries from {args.queries}:")
        for i, q in enumerate(queries, start=1):
            print(f"  {i:3d}. {q}")
        print("\nNothing captured (--dry-run). Drop the flag to run for real.")
        return 0

    print(f"Capturing {len(queries)} queries at top_k={args.top_k}, "
          f"repeat={args.repeat}, rank profile={HYBRID_PROFILE}")
    print("Retrieval only — no LLM validation.\n")

    client = get_vespa_client()
    captured = []
    for i, query in enumerate(queries, start=1):
        print(f"[{i}/{len(queries)}] {query[:70]}")
        try:
            entry = capture_query(client, query, args.top_k, args.top_n, args.repeat)
        except Exception as e:
            print(f"  [ERROR] {e}", file=sys.stderr)
            captured.append({"query": query, "error": str(e)})
            continue
        p = entry["pool"]
        print(
            f"  pool={p['result_count']} "
            f"lanes={p['meta'].get('per_lane_counts')} "
            f"median={statistics.median(p['elapsed_s']):.1f}s"
        )
        captured.append(entry)

    doc = {
        "schema_version": SCHEMA_VERSION,
        "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "top_k": args.top_k,
        "top_n": args.top_n,
        "hybrid_profile": HYBRID_PROFILE,
        "repeat": args.repeat,
        "vespa_memory": vespa_memory(),
        "queries": captured,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"\nWrote {out} ({len(captured)} queries, git {doc['git_sha']})")
    return 0


# ----- compare ---------------------------------------------------------------


def overlap_at(a: list[dict], b: list[dict], k: int) -> float | None:
    """Jaccard overlap of the top-k parent id sets.

    Returns:
        Overlap fraction, or None when neither side has k results.
    """
    sa = {e["parent_id"] for e in a[:k]}
    sb = {e["parent_id"] for e in b[:k]}
    if not sa and not sb:
        return None
    return round(len(sa & sb) / len(sa | sb), 4)


def rank_displacement(a: list[dict], b: list[dict]) -> dict | None:
    """How far ids present in both lists moved."""
    pos_a = {e["parent_id"]: i for i, e in enumerate(a)}
    pos_b = {e["parent_id"]: i for i, e in enumerate(b)}
    shared = set(pos_a) & set(pos_b)
    if not shared:
        return None
    deltas = [abs(pos_a[p] - pos_b[p]) for p in shared]
    return {
        "shared": len(shared),
        "median_move": round(statistics.median(deltas), 2),
        "max_move": max(deltas),
    }


def median_or_none(values) -> float | None:
    return round(statistics.median(values), 3) if values else None


def compare_block(name: str, ba: dict, bb: dict, top_n: int) -> dict:
    ra, rb = ba.get("ranked", []), bb.get("ranked", [])
    gained = [e["parent_id"] for e in rb if e["parent_id"] not in {x["parent_id"] for x in ra}]
    lost = [e["parent_id"] for e in ra if e["parent_id"] not in {x["parent_id"] for x in rb}]
    ta = median_or_none(ba.get("elapsed_s") or [])
    tb = median_or_none(bb.get("elapsed_s") or [])
    return {
        "block": name,
        "overlap@10": overlap_at(ra, rb, 10),
        f"overlap@{top_n}": overlap_at(ra, rb, top_n),
        "displacement": rank_displacement(ra, rb),
        "gained": gained,
        "lost": lost,
        "latency_s": {"a": ta, "b": tb,
                      "delta": round(tb - ta, 3) if (ta is not None and tb is not None) else None},
        "score_stats": {"a": ba.get("score_stats"), "b": bb.get("score_stats")},
    }


def cmd_lanes(args) -> int:
    """Which lanes earn their keep."""
    doc = json.loads(Path(args.snapshot).read_text(encoding="utf-8"))
    depths = sorted({args.page_size, doc.get("top_n", 50)})

    appear: dict[str, int] = {}
    unique: dict[str, int] = {}
    unique_ranks: dict[str, list[int]] = {}
    at_depth: dict[int, dict[str, int]] = {d: {} for d in depths}
    queries = 0

    for q in doc.get("queries", []):
        ranked = (q.get("pool") or {}).get("ranked") or []
        if not ranked:
            continue
        queries += 1
        for rank, entry in enumerate(ranked, start=1):
            lanes = entry.get("lanes") or []
            for lane in lanes:
                appear[lane] = appear.get(lane, 0) + 1
            if len(lanes) == 1:
                lane = lanes[0]
                unique[lane] = unique.get(lane, 0) + 1
                unique_ranks.setdefault(lane, []).append(rank)
                for d in depths:
                    if rank <= d:
                        at_depth[d][lane] = at_depth[d].get(lane, 0) + 1

    if not queries:
        print("No pool rankings with lane attribution in this snapshot.", file=sys.stderr)
        return 1

    lanes = sorted(set(appear) | set(unique))
    print(f"{args.snapshot}  git={doc.get('git_sha')}  profile={doc.get('hybrid_profile','?')}")
    print(f"{queries} queries, top {doc.get('top_n', 50)} of each pool recorded\n")

    head = f"{'lane':<18} {'in top-N':>9} {'only lane':>10} {'med rank':>9}"
    for d in depths:
        head += f" {'uniq@' + str(d):>9}"
    print(head)
    print("-" * len(head))
    for lane in lanes:
        ranks = unique_ranks.get(lane, [])
        med = f"{statistics.median(ranks):.0f}" if ranks else "-"
        row = f"{lane:<18} {appear.get(lane,0):>9} {unique.get(lane,0):>10} {med:>9}"
        for d in depths:
            row += f" {at_depth[d].get(lane,0):>9}"
        print(row)

    print(f"\n`only lane` = results no other lane found. uniq@N = how many of those")
    print(f"landed in the first N, i.e. what the LLM actually validates.")
    print(f"A lane with a high `only lane` count but ~0 uniq@{args.page_size} is paying")
    print("for a query (and possibly an LLM call) to fill the bottom of the pool.")
    return 0


def cmd_compare(args) -> int:
    missing = [p for p in (args.a, args.b) if not Path(p).exists()]
    if missing:
        for p in missing:
            print(f"No such snapshot: {p}", file=sys.stderr)
        print("\nCapture it first:\n"
              f"  {Path(sys.argv[0]).name} capture -q <queries> -o {missing[0]}",
              file=sys.stderr)
        return 1

    try:
        a = json.loads(Path(args.a).read_text(encoding="utf-8"))
        b = json.loads(Path(args.b).read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"Snapshot is not valid JSON ({e}). A capture that was interrupted "
              "part-way leaves a truncated file; re-run it.", file=sys.stderr)
        return 1

    for doc, label in ((a, args.a), (b, args.b)):
        if doc.get("schema_version") != SCHEMA_VERSION:
            print(f"[warn] {label} has schema_version "
                  f"{doc.get('schema_version')}, expected {SCHEMA_VERSION}", file=sys.stderr)

    top_n = min(a.get("top_n", 50), b.get("top_n", 50))
    pa, pb = a.get("hybrid_profile", "?"), b.get("hybrid_profile", "?")
    print(f"A: {args.a}  git={a.get('git_sha')}  profile={pa}  captured={a.get('captured_at')}")
    print(f"B: {args.b}  git={b.get('git_sha')}  profile={pb}  captured={b.get('captured_at')}")
    if pa != pb:
        print(f"\nRank profile differs: {pa} -> {pb}. This is a ranking comparison,\n"
              "so movement is expected; judge it against the noise floor.")
    elif a.get("git_sha") == b.get("git_sha"):
        print("\nSame git sha and rank profile on both sides — noise-floor measurement.")
    print("=" * 78)

    by_query_a = {q["query"]: q for q in a["queries"] if "error" not in q}
    by_query_b = {q["query"]: q for q in b["queries"] if "error" not in q}
    shared_queries = [q for q in by_query_a if q in by_query_b]
    if not shared_queries:
        print("No queries in common.", file=sys.stderr)
        return 1

    agg: dict[str, list[float]] = {}
    per_query_overlap: list[dict] = []
    for query in shared_queries:
        qa, qb = by_query_a[query], by_query_b[query]
        print(f"\n{query}")

        blocks = [("pool", qa["pool"], qb["pool"])]
        for name in ("atomic", "bm25"):
            if name in qa.get("deterministic", {}) and name in qb.get("deterministic", {}):
                blocks.append((f"det.{name}", qa["deterministic"][name], qb["deterministic"][name]))

        for name, ba, bb in blocks:
            c = compare_block(name, ba, bb, top_n)
            ov_n = c[f"overlap@{top_n}"]
            disp = c["displacement"]
            lat = c["latency_s"]
            print(
                f"  {name:<10} overlap@10={_fmt(c['overlap@10'])} "
                f"overlap@{top_n}={_fmt(ov_n)} "
                f"moved(med)={disp['median_move'] if disp else '-'} "
                f"+{len(c['gained'])}/-{len(c['lost'])} "
                f"lat {_fmt(lat['a'])}s→{_fmt(lat['b'])}s"
            )
            if ov_n is not None:
                agg.setdefault(f"{name}:overlap@{top_n}", []).append(ov_n)
                per_query_overlap.append({
                    "block": name, "query": query, "overlap": ov_n,
                    "count_a": ba.get("result_count"), "count_b": bb.get("result_count"),
                    "gained": len(c["gained"]), "lost": len(c["lost"]),
                })
            if lat["delta"] is not None:
                agg.setdefault(f"{name}:latency_delta", []).append(lat["delta"])

        la = (qa["pool"].get("meta") or {}).get("per_lane_counts") or {}
        lb = (qb["pool"].get("meta") or {}).get("per_lane_counts") or {}
        deltas = {k: (lb.get(k, 0) - la.get(k, 0)) for k in sorted(set(la) | set(lb))}
        if any(deltas.values()):
            print(f"  lane count deltas: {deltas}")

    print("\n" + "=" * 78)
    print(f"AGGREGATE over {len(shared_queries)} queries")
    for key in sorted(agg):
        vals = agg[key]
        print(f"  {key:<28} median={statistics.median(vals):.4f}  "
              f"min={min(vals):.4f}  max={max(vals):.4f}")

    # Which queries actually disagree. The aggregate hides this, and knowing
    # *which* query is unstable is the whole diagnostic — a value that repeats
    # exactly across independent comparisons is a reproducible difference, not
    # run-to-run noise.
    print("\n" + "-" * 78)
    print("LEAST STABLE QUERIES (overlap < 1.0)")
    for block in ("pool", "det.atomic", "det.bm25"):
        rows = sorted(
            (v for v in per_query_overlap if v["block"] == block and v["overlap"] < 1.0),
            key=lambda v: v["overlap"],
        )[:5]
        if not rows:
            print(f"  {block:<12} all queries identical")
            continue
        print(f"  {block}")
        for r in rows:
            print(f"    {r['overlap']:.4f}  n={r['count_a']}/{r['count_b']}  "
                  f"+{r['gained']}/-{r['lost']}  {r['query'][:56]!r}")

    det_keys = [k for k in agg if k.startswith("det.") and "overlap" in k]
    if det_keys and all(min(agg[k]) == 1.0 for k in det_keys):
        print("\n  Deterministic strategies are identical across both captures.")
    elif det_keys:
        print("\n  Deterministic strategies DIFFER. They call no LLM, so this is a\n"
              "  real retrieval change, not run-to-run noise.")
    return 0


def _fmt(v) -> str:
    return "-" if v is None else f"{v:.3f}" if isinstance(v, float) else str(v)


# ----- cli -------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Capture and compare retrieval snapshots (no LLM validation).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    cap = sub.add_parser("capture", help="Run the query set and write a snapshot.")
    cap.add_argument("-q", "--queries", required=True,
                     help="Query set: .csv with a 'query' column, or one query per line.")
    cap.add_argument("-o", "--out", required=True, help="Output JSON path.")
    cap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                     help=f"Retrieval breadth (default {DEFAULT_TOP_K}, matches the UI).")
    cap.add_argument("--top-n", type=int, default=50,
                     help="How many ranked ids to record per query (default 50).")
    cap.add_argument("--repeat", type=int, default=3,
                     help="Runs per query, for latency sampling (default 3).")
    cap.add_argument("--dry-run", action="store_true",
                     help="Print the parsed queries and exit, without querying Vespa.")
    cap.add_argument("-v", "--verbose", action="store_true",
                     help="Keep pyvespa's per-request HTTP logging.")
    cap.set_defaults(func=cmd_capture)

    lanes = sub.add_parser("lanes", help="Per-lane contribution within one snapshot.")
    lanes.add_argument("snapshot")
    lanes.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE,
                       help="LLM validation page size (CombinedVespaStrategy.PAGE_SIZE).")
    lanes.set_defaults(func=cmd_lanes)

    cmp_ = sub.add_parser("compare", help="Diff two snapshots.")
    cmp_.add_argument("a", help="Baseline snapshot (before).")
    cmp_.add_argument("b", help="New snapshot (after).")
    cmp_.set_defaults(func=cmd_compare)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
