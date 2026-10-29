# --- A. Imports, settings an loading the data
"""Forecasting and resource planing.
Model: ordinary least squares regression with
linear trend + day-of-week dummies + month dummies.
simple, explainable, and easy to back-test (the same approach used for real budgeting planning in operation teams).
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

# --- B. Turning dates into model inputs ("featurs")
def _features(dates: pd.DatetimeIndex, origin: pd.Timestamp, months: bool = True) -> np.ndarray:
    t = ((dates - origin).days.values / 365.0)[:, None]
    parts = [np.ones((len(dates), 1)), t, np.eye(7)[dates.dayofweek][:, 1:]]
    if months:
        parts.append(np.eye(12)[dates.month - 1][:, 1:])
    return np.hstack(parts)

# --- C. The model and the error measure
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

# --- D. The main forecast functions
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

    #back-test: train without the last N days, then measure the error on thse N days
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