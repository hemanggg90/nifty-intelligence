"""Does a tighter stop help? Backtest every strategy at several STOP_ATR_SCALE values and compare.

    python scripts/evaluate_stop_scales.py                       # all cached instruments, scales 1.0 / 0.75 / 0.5
    python scripts/evaluate_stop_scales.py NIFTY BANKNIFTY --scales 1 0.5 --budget 1800
    python scripts/evaluate_stop_scales.py --json out.json

Uses cached candles only (it never needs a Dhan token), and prices each trade as a bought ATM option with the
Black-Scholes premium model (flat volatility from the bars, no skew) so the stop can be expressed as a % of premium
and as rupees per lot. For every scale it reports, over all strategy trades:
  trades, win rate, mean net R (after charges) with a 95% interval, median stop as % of the option premium, and
  the share of trades whose ONE-LOT stop-loss fits inside the per-trade loss budget (so they could actually be taken).
Tighter stops always raise the share that fit; whether they cost edge is what the net-R columns say. The history is
~60 trading days, so intervals are wide: read a difference smaller than the interval as "no measurable change".
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from quant_intelligence.backtesting import engine as engine_module  # noqa: E402
from quant_intelligence.backtesting.engine import net_r_multiple, run_backtest  # noqa: E402
from quant_intelligence.config import settings as settings_module  # noqa: E402
from quant_intelligence.config.watchlist import COMMODITY_LOT_SIZES, WATCHLIST_COMMODITIES, WATCHLIST_STOCKS  # noqa: E402
from quant_intelligence.data.data_manager import DataManager  # noqa: E402
from quant_intelligence.features.feature_engine import compute_features  # noqa: E402
from quant_intelligence.research.pipeline import _uses_volume  # noqa: E402
from quant_intelligence.strategies.registry import get_all_strategies  # noqa: E402
from quant_intelligence.utils.market_profile import profile_for  # noqa: E402
from quant_intelligence.utils.timeutil import now_ist  # noqa: E402

FALLBACK_LOTS = {"NIFTY": 75, "BANKNIFTY": 30, "FINNIFTY": 60, "MIDCPNIFTY": 120, "SENSEX": 20}


def lot_size(symbol: str) -> int | None:
    if symbol in COMMODITY_LOT_SIZES:
        return COMMODITY_LOT_SIZES[symbol]
    try:
        from quant_intelligence.data_adapters.dhan_instrument_master import resolve_derivative_lot_specs

        spec = resolve_derivative_lot_specs(symbol)
        if spec:
            return int(spec["lot_size"])
    except Exception:
        pass
    return FALLBACK_LOTS.get(symbol)


def set_scale(scale: float) -> None:
    """Point every module that reads the setting at a copy with this scale (and the premium model switched on)."""
    new = dataclasses.replace(settings_module.SETTINGS, stop_atr_scale=scale, vol_premium_model_in_backtest=True)
    settings_module.SETTINGS = new
    engine_module.SETTINGS = new


def load(symbol: str):
    end = now_ist()
    ohlcv, meta = DataManager().get_ohlcv(symbol, "5min", end.replace(hour=0, minute=0) - __import__("datetime").timedelta(days=60), end, quiet=True)
    return ohlcv, compute_features(ohlcv, profile_for(symbol))


def collect(symbols: list[str], scale: float, budget: float) -> list[dict]:
    set_scale(scale)
    rows = []
    for sym in symbols:
        try:
            ohlcv, features = load(sym)
        except Exception as e:
            print(f"  {sym}: no data ({e})", file=sys.stderr)
            continue
        lot = lot_size(sym)
        qty = lot or 1  # one lot: the fixed brokerage per order is a real share of a small premium risk
        has_volume = float(ohlcv["volume"].fillna(0).sum()) > 0
        for strat in get_all_strategies():
            if not has_volume and _uses_volume(strat):
                continue
            try:
                result = run_backtest(strat, ohlcv, features, sym, persist=False, quantity=qty)
            except Exception as e:
                print(f"  {sym}/{strat.name}: {e}", file=sys.stderr)
                continue
            for t in result.trades:
                bs = t.setup.meta.get("bs_premium") if t.setup.meta else None
                if not bs:
                    continue
                rows.append({
                    "symbol": sym, "strategy": strat.name, "net_r": net_r_multiple(t, qty, profile_for(sym).name),
                    "win": 1.0 if t.gross_pnl > 0 else 0.0, "stop_pct": bs["risk"] / bs["entry"] * 100.0,
                    "one_lot_risk": bs["risk"] * lot if lot else float("nan"),
                    "fits": float(lot is not None and bs["risk"] * lot <= budget),
                })
    return rows


def summarise(rows: list[dict]) -> dict:
    n = len(rows)
    if not n:
        return {"trades": 0}
    r = np.array([x["net_r"] for x in rows], float)
    mean = float(r.mean())
    se = float(r.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    one_lot = np.array([x["one_lot_risk"] for x in rows], float)
    return {
        "trades": n, "win_rate": float(np.mean([x["win"] for x in rows]) * 100), "mean_net_r": mean, "ci95": (mean - 1.96 * se, mean + 1.96 * se),
        "median_stop_pct": float(np.median([x["stop_pct"] for x in rows])), "median_one_lot_risk": float(np.nanmedian(one_lot)) if np.isfinite(one_lot).any() else float("nan"),
        "fits_budget_pct": float(np.mean([x["fits"] for x in rows]) * 100),
        "instruments": len({x["symbol"] for x in rows}),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("symbols", nargs="*")
    ap.add_argument("--scales", type=float, nargs="+", default=[1.0, 0.75, 0.5])
    ap.add_argument("--budget", type=float, default=1800.0, help="rupees one stopped trade may lose (one lot must fit)")
    ap.add_argument("--json")
    args = ap.parse_args()
    symbols = args.symbols or (list(settings_module.SETTINGS.option_underlyings) + [s["symbol"] for s in WATCHLIST_STOCKS]
                               + [c["symbol"] for c in WATCHLIST_COMMODITIES])

    out: dict = {}
    print(f"Per-trade loss budget Rs {args.budget:,.0f}; {len(symbols)} instruments; ~60 days of 5-minute candles\n")
    print(f"{'scale':>6} {'trades':>7} {'win%':>6} {'mean net R':>11} {'95% interval':>16} {'stop % prem':>12} {'1-lot risk Rs':>14} {'fits budget':>12}")
    for scale in args.scales:
        rows = collect(symbols, scale, args.budget)
        s = summarise(rows)
        out[str(scale)] = {"overall": s, "by_symbol": {sym: summarise([r for r in rows if r["symbol"] == sym]) for sym in sorted({r["symbol"] for r in rows})},
                           "by_strategy": {st: summarise([r for r in rows if r["strategy"] == st]) for st in sorted({r["strategy"] for r in rows})}}
        if s["trades"]:
            lo, hi = s["ci95"]
            print(f"{scale:>6g} {s['trades']:>7} {s['win_rate']:>5.1f}% {s['mean_net_r']:>+11.3f} {f'({lo:+.2f}, {hi:+.2f})':>16} "
                  f"{s['median_stop_pct']:>11.1f}% {s['median_one_lot_risk']:>14,.0f} {s['fits_budget_pct']:>11.0f}%")
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
