"""Re-stamp cached enrichments as current, without calling the LLM."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import duckdb

from re_source.config import DUCKDB_PATH
from re_source.db_loader import get_all_documents

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("restamp_enrichment")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report what would change without writing.",
    )
    args = parser.parse_args()

    logger.info("loading documents from DuckDB")
    docs = get_all_documents()
    logger.info(f"DuckDB has {len(docs)} documents")

    conn = duckdb.connect(str(DUCKDB_PATH), read_only=args.dry_run)
    try:
        exists = conn.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_name = 'enrichment'"
        ).fetchone()[0]
        if not exists:
            logger.error("no `enrichment` table — nothing to re-stamp")
            return 1

        cached: dict[str, str] = dict(
            conn.execute("SELECT doc_id, source_hash FROM enrichment").fetchall()
        )
        logger.info(f"cached enrichments: {len(cached)}")

        # Partition the corpus against the cache.
        stale = [d for d in docs if d.id in cached and cached[d.id] != d.content_hash]
        current = [d for d in docs if d.id in cached and cached[d.id] == d.content_hash]
        missing = [d for d in docs if d.id not in cached]

        logger.info(f"  already current : {len(current)}")
        logger.info(f"  stale hash      : {len(stale)}  <- re-stamped by this script")
        logger.info(f"  no profile yet  : {len(missing)}  <- left for `enrich --run`")

        if args.dry_run:
            logger.info("--dry-run: no writes made")
            return 0

        if not stale:
            logger.info("nothing to re-stamp")
            return 0

        conn.executemany(
            "UPDATE enrichment SET source_hash = ? WHERE doc_id = ?",
            [(d.content_hash, d.id) for d in stale],
        )
        logger.info(f"re-stamped {len(stale)} rows")
        logger.info(f"next `enrich --run` should enrich {len(missing)} rows")
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
