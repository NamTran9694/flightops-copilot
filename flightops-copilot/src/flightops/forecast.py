"""Forecasting and resource planning.

Model: ordinary least squares regression with
  linear trend + day-of-week dummies + month dummies.
Simple, explainable, and easy to back-test (the same approach used for
real budget planning in operations teams).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .db import connect

METRICS = {"flights", "labor_hours", "hours_per_flight", "meals"}

# Per-metric model settings.
# Volume metrics: all history, yearly (month) seasonality.
# Hours per flight: improvement is slowing down, so use only the recent 180 days
# and no month dummies (one year of months would be confused with the trend).
MODEL_CONFIG = {
    "flights": {"history_days": None, "months": True},
    "meals": {"history_days": None, "months": True},
    "labor_hours": {"history_days": None, "months": True},
    "hours_per_flight": {"history_days": 180, "months": False},
}


def load_daily(db_path=None) -> pd.DataFrame:
    con = connect(db_path)
    try:
        df = pd.read_sql("SELECT * FROM daily_ops ORDER BY ops_date", con, parse_dates=["ops_date"])
    finally:
        con.close()
    return df.set_index("ops_date")


def _features(dates: pd.DatetimeIndex, origin: pd.Timestamp, months: bool = True) -> np.ndarray:
    t = ((dates - origin).days.values / 365.0)[:, None]
    parts = [np.ones((len(dates), 1)), t, np.eye(7)[dates.dayofweek][:, 1:]]   # Monday = baseline
    if months:
        parts.append(np.eye(12)[dates.month - 1][:, 1:])                       # January = baseline
    return np.hstack(parts)


@dataclass
class SeasonalTrendModel:
    months: bool = True
    coef: np.ndarray | None = None
    origin: pd.Timestamp | None = None
    resid_std: float = 0.0

    def fit(self, y: pd.Series) -> "SeasonalTrendModel":
        self.origin = y.index[0]
        X = _features(y.index, self.origin, self.months)
        self.coef, *_ = np.linalg.lstsq(X, y.values, rcond=None)
        self.resid_std = float(np.std(y.values - X @ self.coef))
        return self

    def predict(self, dates: pd.DatetimeIndex) -> np.ndarray:
        return _features(dates, self.origin, self.months) @ self.coef


def mape(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs((actual - pred) / actual)) * 100)


def fit_metric(df: pd.DataFrame, metric: str) -> SeasonalTrendModel:
    cfg = MODEL_CONFIG[metric]
    y = df[metric].astype(float)
    if cfg["history_days"]:
        y = y.iloc[-cfg["history_days"]:]
    return SeasonalTrendModel(months=cfg["months"]).fit(y)


def forecast_metric(metric: str, horizon_days: int = 30, backtest_days: int = 60, db_path=None) -> dict:
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {sorted(METRICS)}")
    horizon_days = int(max(1, min(horizon_days, 365)))
    df = load_daily(db_path)
    y = df[metric].astype(float)

    # back-test: train without the last N days, then measure the error on those N days
    bt = fit_metric(df.iloc[:-backtest_days], metric)
    test = y.iloc[-backtest_days:]
    bt_mape = mape(test.values, bt.predict(test.index))

    model = fit_metric(df, metric)
    future = pd.date_range(y.index[-1] + pd.Timedelta(days=1), periods=horizon_days, freq="D")
    pred = model.predict(future)
    band = 1.96 * model.resid_std

    weekly = pd.Series(pred, index=future).resample("W-SUN").mean().round(2)
    return {
        "metric": metric,
        "last_actual_date": y.index[-1].strftime("%Y-%m-%d"),
        "last_30d_actual_avg": round(float(y.iloc[-30:].mean()), 2),
        "forecast_start": future[0].strftime("%Y-%m-%d"),
        "forecast_end": future[-1].strftime("%Y-%m-%d"),
        "forecast_daily_avg": round(float(pred.mean()), 2),
        "forecast_total": round(float(pred.sum()), 1),
        "approx_95pct_daily_range": [round(float(pred.mean() - band), 2), round(float(pred.mean() + band), 2)],
        "weekly_avg_forecast": {d.strftime("%Y-%m-%d"): v for d, v in weekly.items()},
        "backtest_mape_pct": round(bt_mape, 2),
        "model": "OLS: trend + day-of-week" + (" + month seasonality" if model.months else " (recent 180 days)"),
    }


def plan_resources(start_date: str, end_date: str, db_path=None) -> dict:
    """Labor hours, FTE and trucks needed for a future period (follows budget_planning_process)."""
    df = load_daily(db_path)
    start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
    last = df.index[-1]
    if start <= last:
        raise ValueError(f"start_date must be after the last actual date {last:%Y-%m-%d}.")
    if (end - start).days > 366 or end < start:
        raise ValueError("Period must be between 1 and 366 days.")
    dates = pd.date_range(start, end, freq="D")

    flights = fit_metric(df, "flights").predict(dates)
    hpf_model = fit_metric(df, "hours_per_flight").predict(dates)
    hpf_floor = float(df["hours_per_flight"].iloc[-90:].mean())
    hpf = np.maximum(hpf_model, hpf_floor)   # policy: latest 90-day avg is the floor

    con = connect(db_path)
    try:
        wide_share = con.execute(
            "SELECT AVG(aircraft_type IN ('A330','A350')) FROM flights "
            "WHERE flight_date >= date((SELECT MAX(flight_date) FROM flights), '-90 day')").fetchone()[0]
    finally:
        con.close()

    labor_hours = flights * hpf
    weeks = len(dates) / 7
    fte = labor_hours.sum() / weeks / 40 * 1.12
    nb_equiv = flights * (1 + wide_share)          # widebody counts as 2 narrowbody
    trucks = np.ceil(nb_equiv / 12 * 1.10)

    return {
        "period": f"{start:%Y-%m-%d} to {end:%Y-%m-%d}",
        "days": len(dates),
        "forecast_flights_total": int(flights.sum()),
        "forecast_flights_daily_avg": round(float(flights.mean()), 1),
        "hours_per_flight_used": round(float(hpf.mean()), 2),
        "hours_per_flight_floor_90d": round(hpf_floor, 2),
        "required_labor_hours_total": round(float(labor_hours.sum())),
        "required_fte_incl_12pct_buffer": round(float(fte), 1),
        "widebody_share": round(float(wide_share), 3),
        "trucks_needed_peak_day": int(trucks.max()),
        "trucks_needed_avg_day": round(float(trucks.mean()), 1),
        "assumptions": "Rules from budget_planning_process: FTE = weekly hours / 40 x 1.12; "
                       "trucks = narrowbody-equivalent flights / 12 x 1.10.",
    }
