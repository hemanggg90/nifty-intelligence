"""Persist and read volatility forecasts and model scores (tables `vol_forecasts`, `vol_model_scores`).

Heavy fitting runs once a day in the data keeper; trading cycles and pages only call the cheap read functions
here, so a scan never fits a model.
"""
from __future__ import annotations

import datetime as dt

from quant_intelligence.utils.logging_utils import log_event
from quant_intelligence.volatility.forecast_engine import InstrumentResult


def save_result(result: InstrumentResult) -> bool:
    """Write the forecasts (selected one flagged) and model scores of one run. False if the DB write failed."""
    try:
        from quant_intelligence.database.db import get_session
        from quant_intelligence.database.models import VolForecastRow, VolModelScore, utcnow

        evaluated_at = utcnow()  # one timestamp for the whole run, so latest_scores can find its rows
        with get_session() as session:
            for name, f in result.forecasts.items():
                session.add(VolForecastRow(
                    instrument=result.instrument, model=name, asof=f.asof.to_pydatetime() if f.asof is not None else None,
                    horizon_days=f.horizon_days, sigma_1d=f.sigma_1d, sigma_horizon=f.sigma_horizon,
                    selected=(name == result.selected), diagnostics=_jsonable(f.diagnostics),
                ))
            for row in result.scores:
                session.add(VolModelScore(
                    instrument=result.instrument, model=row["model"], evaluated_at=evaluated_at, n_obs=row["n"], qlike=_num(row["qlike"]),
                    mse=_num(row["mse"]), mz_alpha=_num(row["mz_alpha"]), mz_beta=_num(row["mz_beta"]),
                    mz_r2=_num(row["mz_r2"]), mz_p=_num(row["mz_p"]), dm_stat=_num(row["dm_stat"]),
                    dm_p=_num(row["dm_p"]), selected=bool(row["selected"]),
                    details=_jsonable({"reason": result.reason, **result.diagnostics}),
                ))
        return True
    except Exception as e:
        log_event("vol_store", f"Could not save volatility result for {result.instrument}: {e}", level="WARNING")
        return False


def latest_forecast(instrument: str, max_age_days: float | None = None) -> dict | None:
    """The selected model's most recent stored forecast {model, asof, sigma_1d, sigma_horizon, horizon_days,
    created_at, diagnostics}, or None if there is none (or it is older than `max_age_days`)."""
    try:
        from quant_intelligence.database.db import get_session
        from quant_intelligence.database.models import VolForecastRow

        with get_session() as session:
            row = (session.query(VolForecastRow).filter_by(instrument=instrument, selected=True)
                   .order_by(VolForecastRow.id.desc()).first())
            if row is None:
                return None
            out = {"model": row.model, "asof": row.asof, "sigma_1d": row.sigma_1d, "sigma_horizon": row.sigma_horizon,
                   "horizon_days": row.horizon_days, "created_at": row.created_at, "diagnostics": row.diagnostics or {}}
        if max_age_days is not None and out["created_at"] is not None:
            age = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None) - out["created_at"]
            if age > dt.timedelta(days=max_age_days):
                return None
        return out
    except Exception:
        return None


def latest_scores(instrument: str) -> list[dict]:
    """The most recent evaluation's score rows for an instrument (one per model)."""
    try:
        from quant_intelligence.database.db import get_session
        from quant_intelligence.database.models import VolModelScore

        with get_session() as session:
            newest = (session.query(VolModelScore).filter_by(instrument=instrument)
                      .order_by(VolModelScore.id.desc()).first())
            if newest is None:
                return []
            rows = (session.query(VolModelScore).filter_by(instrument=instrument, evaluated_at=newest.evaluated_at)
                    .order_by(VolModelScore.qlike).all())
            return [{"model": r.model, "n": r.n_obs, "qlike": r.qlike, "mse": r.mse, "mz_alpha": r.mz_alpha,
                     "mz_beta": r.mz_beta, "mz_r2": r.mz_r2, "mz_p": r.mz_p, "dm_stat": r.dm_stat, "dm_p": r.dm_p,
                     "selected": r.selected, "evaluated_at": r.evaluated_at} for r in rows]
    except Exception:
        return []


def _num(x) -> float | None:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if x == x and abs(x) != float("inf") else None


def _jsonable(obj):
    """JSON-safe copy: NaN/inf -> None, numpy/pandas scalars -> python, unknown objects -> str."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (str, bool)) or obj is None:
        return obj
    if hasattr(obj, "item"):
        obj = obj.item()
    if isinstance(obj, (int, float)):
        return _num(obj) if isinstance(obj, float) else obj
    return str(obj)
