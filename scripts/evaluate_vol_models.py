"""Out-of-sample comparison of the volatility models (EWMA, GARCH, GJR-GARCH, EGARCH, HAR-RV).

    python scripts/evaluate_vol_models.py NIFTY BANKNIFTY
    python scripts/evaluate_vol_models.py NIFTY --years 8 --models EWMA GARCH HAR-RV
    python scripts/evaluate_vol_models.py RELIANCE --csv-only        # never call Dhan, use files only

Daily history comes from data_cache/csv/<SYMBOL>_1d.csv, the parquet cache, then Dhan (needs a valid token).
Prints, per instrument, each model's out-of-sample QLIKE (lower is better), MSE, the Mincer-Zarnowitz fit and the
Diebold-Mariano test against EWMA, and which model the selection rule picks. EWMA is the default: another
model is selected only if it beats EWMA significantly. Nothing is written to the database.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quant_intelligence.config.settings import SETTINGS  # noqa: E402
from quant_intelligence.data.daily_history import load_daily_history  # noqa: E402
from quant_intelligence.volatility.forecast_engine import DEFAULT_MODELS, run_instrument  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("symbols", nargs="+")
    ap.add_argument("--years", type=int, default=SETTINGS.vol_history_years)
    ap.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    ap.add_argument("--refit-every", type=int, default=SETTINGS.vol_refit_every_days)
    ap.add_argument("--min-train", type=int, default=SETTINGS.vol_min_train_days)
    ap.add_argument("--min-eval", type=int, default=SETTINGS.vol_min_eval_days)
    ap.add_argument("--alpha", type=float, default=SETTINGS.vol_dm_alpha)
    ap.add_argument("--csv-only", action="store_true", help="do not call Dhan; use CSV/cache files only")
    args = ap.parse_args()

    status = 0
    for sym in args.symbols:
        daily, info = load_daily_history(sym.upper(), years=args.years, allow_fetch=not args.csv_only)
        print(f"\n=== {sym.upper()}  {info['n']} daily bars {info.get('first')} -> {info.get('last')}  (source: {info['source']})")
        if info.get("fetch_error"):
            print(f"    Dhan: {info['fetch_error']}")
        result = run_instrument(sym.upper(), daily, models=tuple(args.models), refit_every=args.refit_every,
                                min_train=args.min_train, min_eval=args.min_eval, alpha=args.alpha)
        if not result.scores:
            print(f"    {result.reason}")
            if result.forecast is not None:
                f = result.forecast
                print(f"    EWMA only: 1-day vol {f.sigma_1d:.2%} (annualised {f.sigma_1d * (SETTINGS.trading_days_per_year ** 0.5):.1%})")
            status = status or 2
            continue
        print(f"    {'model':<8} {'n':>5} {'QLIKE':>9} {'MSE(x1e8)':>10} {'MZ a':>9} {'MZ b':>7} {'MZ R2':>6} {'DM p vs EWMA':>13}")
        for r in sorted(result.scores, key=lambda r: r["qlike"]):
            dm = "-" if r["dm_p"] != r["dm_p"] else f"{r['dm_p']:.3f}"
            mark = "  <- selected" if r["selected"] else ""
            print(f"    {r['model']:<8} {r['n']:>5} {r['qlike']:>9.4f} {r['mse'] * 1e8:>10.3f} {r['mz_alpha']:>9.2e} "
                  f"{r['mz_beta']:>7.2f} {r['mz_r2']:>6.2f} {dm:>13}{mark}")
        fb = result.diagnostics.get("fallbacks", {})
        if any(fb.values()):
            print(f"    fallbacks to EWMA (days): {fb}")
        if result.diagnostics.get("errors"):
            print(f"    model errors: {result.diagnostics['errors']}")
        print(f"    -> {result.reason}")
        f = result.forecast
        if f is not None:
            print(f"    latest {f.model} forecast ({f.asof:%Y-%m-%d}): 1-day vol {f.sigma_1d:.2%}, "
                  f"annualised {f.sigma_1d * (SETTINGS.trading_days_per_year ** 0.5):.1%}")
    return status


if __name__ == "__main__":
    sys.exit(main())
