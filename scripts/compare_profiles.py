#!/usr/bin/env python3
"""Show the top results for one query under two rank profiles, side by side."""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.getLogger("httpr").setLevel(logging.WARNING)

from scripts.capture_baseline import load_queries


def run(settings: dict[str, str], query: str, top_k: int, top_n: int) -> list[dict]:
    """Retrieve with `settings` applied to the environment."""
    os.environ.update(settings)
    import re_source.vespa_strategies as vs
    vs = importlib.reload(vs)
    from re_source.vespa_client import get_vespa_client

    result = vs.AtomicVespaStrategy(get_vespa_client()).execute(query, top_k=top_k)
    out = []
    for r in result.results[:top_n]:
        chunks = (r.payload or {}).get("matching_chunks") or []
        best = max(chunks, key=lambda c: c.get("score") or 0, default={})
        out.append({
            "parent_id": r.doc_id,
            "score": r.score,
            "column": best.get("column_name") or "",
            "text": (best.get("text") or "").replace("\n", " ").strip(),
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("query", help="Query text, or a substring matching one in queries.csv.")
    ap.add_argument("--profiles", nargs=2,
                    help="Two rank profiles to compare (shorthand for --variants).")
    ap.add_argument("--variants", nargs=2, metavar="SPEC",
                    help="Two env specs to compare, e.g. "
                         "VESPA_LEXICAL_STOPWORDS=1 VESPA_LEXICAL_STOPWORDS=0. "
                         "Comma-separate multiple settings within one spec.")
    ap.add_argument("-n", "--top-n", type=int, default=10)
    ap.add_argument("--top-k", type=int, default=500)
    ap.add_argument("--queries", default="queries.csv")
    ap.add_argument("--width", type=int, default=100)
    args = ap.parse_args()

    query = args.query
    qfile = Path(args.queries)
    if qfile.exists():
        matches = [q for q in load_queries(qfile) if args.query.lower() in q.lower()]
        if len(matches) == 1:
            query = matches[0]
            print(f"(matched from {qfile}: {query!r})\n")
        elif len(matches) > 1:
            print(f"{args.query!r} matches {len(matches)} queries in {qfile}:", file=sys.stderr)
            for m in matches:
                print(f"  {m}", file=sys.stderr)
            return 1

    if args.variants:
        specs = []
        for spec in args.variants:
            settings = dict(kv.split("=", 1) for kv in spec.split(","))
            specs.append((spec, settings))
    elif args.profiles:
        specs = [(p, {"VESPA_HYBRID_PROFILE": p}) for p in args.profiles]
    else:
        specs = [("hybrid", {"VESPA_HYBRID_PROFILE": "hybrid"}),
                 ("hybrid_sat", {"VESPA_HYBRID_PROFILE": "hybrid_sat"})]

    labels = [label for label, _ in specs]
    results = {label: run(settings, query, args.top_k, args.top_n)
               for label, settings in specs}

    a, b = labels
    ids_a = [r["parent_id"] for r in results[a]]
    ids_b = [r["parent_id"] for r in results[b]]
    shared = set(ids_a) & set(ids_b)

    print("=" * args.width)
    print(textwrap.fill(f"QUERY: {query}", args.width))
    print(f"top {args.top_n}: {len(shared)} of {len(ids_a)} shared between {a} and {b}")

    for profile in labels:
        other = set(ids_b if profile == a else ids_a)
        print("\n" + "=" * args.width)
        print(f"{profile}")
        print("=" * args.width)
        for i, r in enumerate(results[profile], start=1):
            mark = " " if r["parent_id"] in other else "*"
            print(f"{mark}{i:>3}. [{r['score']:.2f}] {r['column'][:60]}")
            for line in textwrap.wrap(r["text"][:400], args.width - 8) or [""]:
                print(f"        {line}")
    print("\n* = appears only in this profile's top results")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
