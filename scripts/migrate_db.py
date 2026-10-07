"""Copy trade history from a local SQLite file into the durable database named by DATABASE_URL.

    # point DATABASE_URL at the new Postgres (in .env or the shell), then:
    python scripts/migrate_db.py                      # DRY RUN: shows what would be copied, changes nothing
    python scripts/migrate_db.py --apply              # copy it
    python scripts/migrate_db.py --source other.db    # a different SQLite file (default data_cache/quant_intelligence.db)

Copies orders, positions, fills, risk events and stored daily reports. Rows already in the target are skipped
(orders/positions by their ids, reports by date + scope), so it is safe to run again. The fake "Momentum" trades
that early test runs left in the local database (see scripts/purge_test_trades.py) are NOT copied. Research
tables (backtests, observations) and logs stay behind: nothing reads them back.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import create_engine, text  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = ROOT / "data_cache" / "quant_intelligence.db"

# The signature of the test fixtures (same as scripts/purge_test_trades.py).
_FIXTURE_POSITION = "strategy_name = 'Momentum' AND security_id IS NULL AND entry_price BETWEEN 99 AND 101"
_FIXTURE_ORDER = "strategy_name = 'Momentum' AND security_id IS NULL AND price BETWEEN 99 AND 101"


def _coerce(model, row: dict) -> dict:
    """Raw SQLite returns timestamps, dates, JSON and booleans as text/ints; give each column its model's type.
    Columns the source lacks (an older database) are simply absent and take their defaults."""
    import datetime as dt
    import json

    import pandas as pd
    from sqlalchemy import JSON, Boolean, Date, DateTime

    out = {}
    for col in model.__table__.columns:
        if col.name == "id" or col.name not in row:
            continue
        v = row[col.name]
        if v is not None:
            if isinstance(col.type, DateTime) and isinstance(v, str):
                v = pd.to_datetime(v).to_pydatetime()
            elif isinstance(col.type, Date) and isinstance(v, str):
                v = dt.date.fromisoformat(v[:10])
            elif isinstance(col.type, JSON) and isinstance(v, (str, bytes)):
                v = json.loads(v)
            elif isinstance(col.type, Boolean) and isinstance(v, int):
                v = bool(v)
        out[col.name] = v
    return out


def _rows(conn, table: str, model, where: str = "1=1") -> list[dict]:
    try:
        raw = [dict(r._mapping) for r in conn.execute(text(f"SELECT * FROM {table} WHERE {where}"))]
    except Exception:  # the source has no such table (an older database)
        return []
    return [_coerce(model, r) for r in raw]


def migrate(source_url: str, target_engine, apply: bool = False) -> dict:
    """Copy history from `source_url` to `target_engine`. Returns {table: {"found": n, "new": n}}."""
    from quant_intelligence.database import db as db_module
    from quant_intelligence.database.models import DailyReport, Fill, Order, Position, RiskEvent

    source = create_engine(source_url)
    if apply:
        db_module.init_db(target_engine)  # tables and any missing columns on the target (a dry run changes nothing)
    make_session = sessionmaker(bind=target_engine, expire_on_commit=False)
    with source.connect() as src, make_session() as session:
        def existing(query):
            try:
                return set(query())
            except Exception:  # a dry run against a database that has no tables yet: nothing is stored
                session.rollback()
                return set()

        have_orders = existing(lambda: (o for (o,) in session.query(Order.order_id).all()))
        have_positions = existing(lambda: (p for (p,) in session.query(Position.position_id).all()))
        have_reports = existing(lambda: ((d, sc) for d, sc in session.query(DailyReport.report_date, DailyReport.scope).all()))
        have_risk = existing(lambda: ((r.decision_id, r.timestamp) for r in session.query(RiskEvent).all()))

        orders = _rows(src, "orders", Order, f"NOT ({_FIXTURE_ORDER})")
        new_orders = [r for r in orders if r["order_id"] not in have_orders]
        positions = _rows(src, "positions", Position, f"NOT ({_FIXTURE_POSITION})")
        new_positions = [r for r in positions if r["position_id"] not in have_positions]
        new_ids = {r["order_id"] for r in new_orders}
        fills = [r for r in _rows(src, "fills", Fill) if r["order_id"] in new_ids]  # only fills of orders being copied
        risk = _rows(src, "risk_events", RiskEvent)
        new_risk = [r for r in risk if (r.get("decision_id"), r.get("timestamp")) not in have_risk]
        reports = _rows(src, "daily_reports", DailyReport)
        new_reports = [r for r in reports if (r["report_date"], r["scope"]) not in have_reports]

        report = {"orders": {"found": len(orders), "new": len(new_orders)}, "positions": {"found": len(positions), "new": len(new_positions)},
                  "fills": {"found": len(fills), "new": len(fills)}, "risk_events": {"found": len(risk), "new": len(new_risk)},
                  "daily_reports": {"found": len(reports), "new": len(new_reports)}}
        if not apply:
            return report

        # parents first: a fill's order must exist
        session.add_all(Order(**r) for r in new_orders)
        session.add_all(Position(**r) for r in new_positions)
        session.flush()
        session.add_all(Fill(**r) for r in fills)
        session.add_all(RiskEvent(**r) for r in new_risk)
        session.add_all(DailyReport(**r) for r in new_reports)
        session.commit()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="SQLite file to copy from")
    parser.add_argument("--apply", action="store_true", help="actually copy (default is a dry run)")
    args = parser.parse_args()

    from quant_intelligence.database.db import backend_name, get_engine

    if not args.source.exists():
        print(f"No source database at {args.source}")
        return 1
    target = get_engine()
    if str(target.url.database or "").replace("\\", "/") == str(args.source).replace("\\", "/"):
        print("DATABASE_URL points at the source file itself - set it to the new (Postgres) database first.")
        return 1
    print(f"Source: {args.source}\nTarget: {backend_name()} ({target.url.render_as_string(hide_password=True)})")
    result = migrate("sqlite:///" + args.source.as_posix(), target, apply=args.apply)
    for table, n in result.items():
        print(f"  {table:<14} {n['found']:>7} found, {n['new']:>7} new")
    print("\nCopied." if args.apply else "\nDry run - nothing copied. Re-run with --apply.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
