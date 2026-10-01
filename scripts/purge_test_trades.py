"""Remove leftover test-fixture trades from the dashboard database.

Early test runs wrote fake paper trades into the real database (strategy "Momentum" on NIFTY at a premium of
~100 with no option contract), which then look like real history on the Positions page. This script finds
exactly that signature and nothing else.

    python scripts/purge_test_trades.py            # DRY RUN: lists what would be removed, changes nothing
    python scripts/purge_test_trades.py --apply    # backs the database up first, then deletes

A real trade always carries an option contract (security_id) and an actual option premium, so the signature
below cannot match one. Anything ambiguous is only reported, never deleted.
"""
from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = ROOT / "data_cache" / "quant_intelligence.db"

# Signature of the test fixtures (tests/conftest.py builds "Momentum" trades on a flat ~100 price).
POSITIONS_WHERE = "strategy_name = 'Momentum' AND security_id IS NULL AND entry_price BETWEEN 99 AND 101"
ORDERS_WHERE = "strategy_name = 'Momentum' AND security_id IS NULL AND price BETWEEN 99 AND 101"
FILLS_WHERE = f"order_id IN (SELECT order_id FROM orders WHERE {ORDERS_WHERE})"


def count(conn: sqlite3.Connection, table: str, where: str) -> int:
    return conn.execute(f"SELECT COUNT(*) FROM {table} WHERE {where}").fetchone()[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="SQLite file (default: data_cache/quant_intelligence.db)")
    parser.add_argument("--apply", action="store_true", help="actually delete (a backup copy is made first)")
    args = parser.parse_args()

    if not args.db.exists():
        print(f"No database at {args.db}")
        return 1
    conn = sqlite3.connect(args.db)
    plan = [("fills", FILLS_WHERE), ("orders", ORDERS_WHERE), ("positions", POSITIONS_WHERE)]
    counts = {t: count(conn, t, w) for t, w in plan}
    total_pos = conn.execute("SELECT COUNT(*) FROM positions").fetchone()[0]
    print(f"Database: {args.db}")
    for table, n in counts.items():
        print(f"  {table:<10} {n:>5} test-fixture rows")
    print(f"  positions that stay: {total_pos - counts['positions']}")
    others = conn.execute(
        f"SELECT instrument, strategy_name, COUNT(*) FROM positions WHERE NOT ({POSITIONS_WHERE}) GROUP BY 1, 2 ORDER BY 3 DESC"
    ).fetchall()
    if others:
        print("  kept (do not match the fixture signature):")
        for instrument, strategy, n in others:
            print(f"    {instrument:<12} {strategy:<32} x{n}")

    if not any(counts.values()):
        print("Nothing to remove.")
        return 0
    if not args.apply:
        print("\nDry run - nothing changed. Re-run with --apply to delete (a backup is made first).")
        return 0

    backup = args.db.with_name(f"{args.db.stem}.backup-{datetime.now():%Y%m%d-%H%M%S}{args.db.suffix}")
    conn.close()
    shutil.copy2(args.db, backup)
    print(f"\nBackup written: {backup}")
    conn = sqlite3.connect(args.db)
    with conn:  # one transaction: all or nothing
        for table, where in plan:
            conn.execute(f"DELETE FROM {table} WHERE {where}")
    print("Deleted: " + ", ".join(f"{n} {t}" for t, n in counts.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
