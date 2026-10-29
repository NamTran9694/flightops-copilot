"""Try the forecasting model by hand: python scripts/try_forecast.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flightops.forecast import forecast_metric  # noqa: E402

for metric in ["flights", "hours_per_flight", "labor_hours"]:
    f = forecast_metric(metric, horizon_days=30)
    print(f"\n=== {metric} ===")
    print(f"  model:                {f['model']}")
    print(f"  last 30 days (actual): {f['last_30d_actual_avg']}")
    print(f"  next 30 days (forecast): {f['forecast_daily_avg']}  "
          f"range {f['approx_95pct_daily_range']}")
    print(f"  back-test error (MAPE): {f['backtest_mape_pct']}%")

print("\nWeekly flights forecast:")
for week, value in forecast_metric("flights", 30)["weekly_avg_forecast"].items():
    print(f"  week ending {week}: {value} flights/day")