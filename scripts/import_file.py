"""CLI wrapper: import a CSV or XLSX file into DuckDB."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from re_source.importer import import_file, sanitize_table_name  # noqa: E402,F401

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("import_file")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", type=Path, help="Path to a .csv, .xlsx, or .xls file.")
    parser.add_argument(
        "--id-column", required=True,
        help='Column name holding the stable per-row ID. Becomes "_record_id" in DuckDB.',
    )
    parser.add_argument(
        "--table-name",
        help="DuckDB table name. Defaults to a sanitized version of the filename stem.",
    )
    parser.add_argument(
        "--sheet",
        help="Sheet name for XLSX files. Defaults to the first sheet.",
    )
    args = parser.parse_args()

    if not args.path.exists():
        log.error(f"file not found: {args.path}")
        return 1

    stats = import_file(
        path=args.path,
        id_column=args.id_column,
        table_name=args.table_name,
        sheet=args.sheet,
    )
    log.info(f"done: {stats}")
    log.info("next: python -m re_source.embed_vespa --incremental")
    return 0


if __name__ == "__main__":
    sys.exit(main())
