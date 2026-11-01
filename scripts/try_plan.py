"""Try the resource planner by hand: python scripts/try_plan.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flightops.forecast import plan_resources  # noqa: E402

for start, end in [("2026-11-01", "2026-11-30"), ("2026-12-01", "2026-12-31")]:
    p = plan_resources(start, end)
    print(f"\n=== Plan for {p['period']} ({p['days']} days) ===")
    print(f"  Flights (total / per day):   {p['forecast_flights_total']:,} / {p['forecast_flights_daily_avg']}")
    print(f"  Hours per flight used:       {p['hours_per_flight_used']}  (90-day floor {p['hours_per_flight_floor_90d']})")
    print(f"  Labor hours needed:          {p['required_labor_hours_total']:,}")
    print(f"  FTE needed (+12% buffer):    {p['required_fte_incl_12pct_buffer']}")
    print(f"  Trucks (peak day / average): {p['trucks_needed_peak_day']} / {p['trucks_needed_avg_day']}")
    print(f"  Widebody share:              {p['widebody_share']:.1%}")

try:
    plan_resources("2026-01-01", "2026-01-31")
except ValueError as e:
    print("\nBlocked as expected:", e)