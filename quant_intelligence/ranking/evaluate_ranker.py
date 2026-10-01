"""Walk-forward evaluation of the strategy ranker.

Question answered: when the ranker picks strategy S at bar t and S has a live setup at t
(exactly what the auto trader acts on), how good is the NET-of-cost result compared with
just taking a strategy at random among those with a setup at t - and did NO TRADE avoid
bad trades?

No look-ahead: each strategy is back-tested once over the whole series, but the ranker at bar t
only sees trades that had already CLOSED by t (exit <= t) and were entered before t. It goes
through the same code as the live pipeline (research/ranking_core.py).

Run:  python -m quant_intelligence.ranking.evaluate_ranker [SYMBOL ...] [--json out.json]
Uses the real candles already cached in data_cache/parquet_cache (no network).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from quant_intelligence.backtesting.engine import max_drawdown_in_r, net_r_multiple, run_backtest
from quant_intelligence.config.settings import DATA_CACHE_DIR
from quant_intelligence.features.feature_engine import compute_features
from quant_intelligence.ranking.ranking_engine import rank_and_select
from quant_intelligence.research.pipeline import _uses_volume
from quant_intelligence.research.ranking_core import assess_observations, observations_from_trades
from quant_intelligence.strategies.registry import get_all_strategies
from quant_intelligence.utils.market_profile import profile_for


@dataclass
class Decision:
    instrument: str
    timestamp: pd.Timestamp
    day: str
    selected: str | None  # None = NO TRADE
    n_signals: int  # strategies with a setup at this bar
    selected_net_r: float | None  # net R of the selected strategy's trade at this bar (if it had one)
    pick_net_r: float  # mean net R over all strategies' trades at this bar (= picking one at random)


def evaluate_instrument(
    ohlcv: pd.DataFrame,
    instrument: str,
    quantity: int = 50,
    warmup_frac: float = 0.4,
    min_history_bars: int = 400,
    strategies=None,
    max_points: int | None = None,
) -> list[Decision]:
    ohlcv = ohlcv.sort_values("timestamp").reset_index(drop=True)
    features = compute_features(ohlcv, profile_for(instrument))
    feat_by_ts = features.set_index("timestamp")
    ts_to_idx = {ts: i for i, ts in enumerate(ohlcv["timestamp"])}

    strategies = strategies if strategies is not None else get_all_strategies()
    if float(ohlcv["volume"].fillna(0).sum()) <= 0:  # same rule as the live pipeline
        strategies = [s for s in strategies if not _uses_volume(s)]

    per_strategy = {}
    for strat in strategies:
        bt = run_backtest(strat, ohlcv, features, instrument, split="EVAL", quantity=quantity, persist=False)
        # observations_from_trades skips trades whose entry ts has no feature row; keep aligned.
        trades = [t for t in bt.trades if t.setup.timestamp in feat_by_ts.index]
        if not trades:
            continue  # never produces a setup -> permanently INSUFFICIENT_DATA, cannot affect selection
        obs = observations_from_trades(trades, feat_by_ts, strat.name, quantity)
        entry_idx = np.array([ts_to_idx[t.setup.timestamp] for t in trades])
        exit_idx = np.array([ts_to_idx.get(t.exit_timestamp, len(ohlcv)) for t in trades])
        obs["entry_idx"], obs["exit_idx"] = entry_idx, exit_idx
        per_strategy[strat.name] = {
            "obs": obs,
            "entry_idx": entry_idx,
            "exit_idx": exit_idx,
            "net_r_arr": np.array([net_r_multiple(t, quantity) for t in trades]),
            "net_r": {int(ts_to_idx[t.setup.timestamp]): net_r_multiple(t, quantity) for t in trades},
        }

    start_idx = max(int(len(ohlcv) * warmup_frac), min_history_bars)
    signal_bars = sorted({i for d in per_strategy.values() for i in d["net_r"] if i >= start_idx})
    if max_points and len(signal_bars) > max_points:
        signal_bars = signal_bars[:: int(np.ceil(len(signal_bars) / max_points))]

    decisions: list[Decision] = []
    for t in signal_bars:
        current = features.iloc[t].drop(labels=["timestamp"]).to_dict()
        scores = []
        for name, d in per_strategy.items():
            known_mask = (d["exit_idx"] <= t) & (d["entry_idx"] < t)
            metrics = {"max_drawdown_r": max_drawdown_in_r(d["net_r_arr"][known_mask])}
            scores.append(assess_observations(name, d["obs"][known_mask], current, metrics).score)
        ranking = rank_and_select(scores, "OK")

        at_t = {n: d["net_r"][t] for n, d in per_strategy.items() if t in d["net_r"]}
        selected = None if ranking.is_no_trade else ranking.selected_strategy
        decisions.append(
            Decision(
                instrument=instrument,
                timestamp=ohlcv["timestamp"].iloc[t],
                day=str(ohlcv["timestamp"].iloc[t].date()),
                selected=selected,
                n_signals=len(at_t),
                selected_net_r=at_t.get(selected) if selected else None,
                pick_net_r=float(np.mean(list(at_t.values()))),
            )
        )
    return decisions


def _bootstrap_ci(values: np.ndarray, days: np.ndarray, n_boot: int = 2000, seed: int = 7):
    """95% CI of the mean, resampling whole trading days (trades within a day are correlated)."""
    if len(values) == 0:
        return None
    rng = np.random.default_rng(seed)
    uniq = np.unique(days)
    by_day = {d: values[days == d] for d in uniq}
    means = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        means.append(np.concatenate([by_day[d] for d in pick]).mean())
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize(decisions: list[Decision]) -> dict:
    df = pd.DataFrame([asdict(d) for d in decisions])
    if df.empty:
        return {"decision_points": 0}
    traded = df[df["selected_net_r"].notna()]
    no_trade = df[df["selected"].isna()]
    waiting = df[df["selected"].notna() & df["selected_net_r"].isna()]  # picked S, but S had no setup at t

    out = {
        "decision_points": int(len(df)),
        "no_trade_points": int(len(no_trade)),
        "selected_but_no_setup": int(len(waiting)),
        "trades_taken": int(len(traded)),
        "avg_signals_per_point": float(df["n_signals"].mean()),
    }
    if len(traded):
        sel, pick = traded["selected_net_r"].to_numpy(), traded["pick_net_r"].to_numpy()
        days = traded["day"].to_numpy()
        out.update(
            {
                "selected_mean_net_r": float(sel.mean()),
                "selected_win_rate": float((sel > 0).mean()),
                "random_pick_mean_net_r": float(pick.mean()),
                "lift_vs_random_pick": float((sel - pick).mean()),
                "selected_mean_net_r_ci95": _bootstrap_ci(sel, days),
                "lift_ci95": _bootstrap_ci(sel - pick, days),
            }
        )
    if len(no_trade):
        out["avoided_by_no_trade_mean_net_r"] = float(no_trade["pick_net_r"].mean())
    out["take_every_signal_mean_net_r"] = float(df["pick_net_r"].mean())
    return out


def load_cached_ohlcv(symbol: str, timeframe: str = "5min") -> pd.DataFrame | None:
    path = DATA_CACHE_DIR / "parquet_cache" / f"{symbol}_{timeframe}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def evaluate_symbols(symbols: list[str], timeframe: str = "5min", **kwargs) -> tuple[dict, list[Decision]]:
    all_decisions: list[Decision] = []
    per_symbol = {}
    for sym in symbols:
        df = load_cached_ohlcv(sym, timeframe)
        if df is None or len(df) < 600:
            per_symbol[sym] = {"skipped": "no cached data"}
            continue
        d = evaluate_instrument(df, sym, **kwargs)
        per_symbol[sym] = summarize(d)
        all_decisions += d
    return {"per_symbol": per_symbol, "overall": summarize(all_decisions)}, all_decisions


def main() -> None:
    from quant_intelligence.config.watchlist import WATCHLIST_STOCKS

    ap = argparse.ArgumentParser()
    ap.add_argument("symbols", nargs="*")
    ap.add_argument("--json")
    ap.add_argument("--max-points", type=int, default=None, help="cap decision points per instrument (speed)")
    args = ap.parse_args()
    symbols = args.symbols or ["NIFTY", "BANKNIFTY"] + [s["symbol"] for s in WATCHLIST_STOCKS]

    report, _ = evaluate_symbols(symbols, max_points=args.max_points)
    for sym, r in report["per_symbol"].items():
        print(f"{sym:11}", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if not k.endswith("ci95")})
    print("\nOVERALL")
    for k, v in report["overall"].items():
        print(f"  {k:32}", tuple(round(x, 3) for x in v) if isinstance(v, tuple) else (round(v, 3) if isinstance(v, float) else v))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)


if __name__ == "__main__":
    main()
