"""Tests for forecasting. Run with: pytest -v"""
import numpy as np
import pandas as pd
import pytest

from flightops import forecast


def test_mape_simple_example():
    # actual 100, predicted 90 -> 10% error
    assert forecast.mape(np.array([100.0]), np.array([90.0])) == pytest.approx(10.0)


def test_model_learns_a_perfect_trend():
    dates = pd.date_range("2025-01-01", periods=400, freq="D")
    y = pd.Series(100 + 0.5 * np.arange(400), index=dates)      # straight line, no noise
    model = forecast.SeasonalTrendModel().fit(y)
    future = pd.date_range("2026-02-05", periods=3, freq="D")
    expected = 100 + 0.5 * np.arange(400, 403)
    assert np.allclose(model.predict(future), expected)


def test_flight_forecast_backtest_error_is_small():
    f = forecast.forecast_metric("flights", horizon_days=30)
    assert f["backtest_mape_pct"] < 5
    assert len(f["weekly_avg_forecast"]) >= 4


def test_forecast_rejects_unknown_metric():
    with pytest.raises(ValueError):
        forecast.forecast_metric("revenue", 30)