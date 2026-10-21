"""Generate a realistic SYNTHETIC dataset for an airline catering facility."""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "data" / "flightops.db"

DESTINATIONS = ["LAX", "JFK", "LGA", "BOS", "MCO", "DFW", "ORD", "SEA", "SLC", "MSP",
                "DTW", "LAS", "PHX", "DEN", "MIA", "FLL", "TPA", "SFO", "CDG", "AMS",
                "LHR", "FCO", "ICN", "NRT", "CUN", "SJU", "BWI", "DCA", "CLT", "RDU"]
INTERNATIONAL = {"CDG", "AMS", "LHR", "FCO", "ICN", "NRT"}
AIRCAFT = { #aircaft type -> (share of flights, average meals)
"A321": (0.33, 160), "B737-900": (0.28, 150), "A220": (0.17, 90),
"B757": (0.14, 170), "A330": (0.05, 260), "A350": (0.03, 300),
}

# How many flights each day?
def daily_flight_counts(dates: pd.DatetimeIndex, rng:np.ramdom.Generator) -> np.ndarray:
    """Number of flights per day = trend x weekday x month x noise"""
    t = np.linspace(0, 1, len(dates))
    base = 600 + 300 * t
    dow = np.array([1.03, 1.0, 0.97, 1.0, 1.05, 0.9, 1.02])[dates.dayofweek]
    month = np.array([0.94, 0.92, 1.0, 1.0, 1.03, 1.08, 1.10, 1.07, 0.97, 0.99, 1.0, 1.04])[dates.month - 1]
    noise = rng.normal(1.0, 0.025, len(dates))
    return np.round(base * dow * month * noise).astype(int)

# Labor Hours per Flights (KPI)
def hours_per_flight(dates: pd.DatetimeIndex, rng: np.random.Generator) -> np.ndarray:
    """Labor hours per flight: improves fast at first, then slows down (19.0 -> 15.6)."""
    t = np.linspace(0, 1, len(dates))
    hpf = 19.0 -3.4*(1 - np.exp(-2.2 * t)) / (1 - np.exp(-2.2))
    hpf *= np.where(dates.dayofweek == 5, 1.04, 1.0)
    return hpf * rng.normal(1.0, 0.03, len(dates))

# Build one row per flight
def build_flights(dates, counts, rng) -> pd.DataFrame:
    """One row per flight."""
    n = counts.sum()
    flight_date = np.repeat(dates.values, counts)
    types = list(AIRCAFT)
    ac = rng.choice(types, size=n, p=[AIRCAFT[a][0] for a in types])
    w = np.array([0.35 if d in INTERNATIONAL else 1.0 for d in DESTINATIONS])
    dest = rng.choice(DESTINATIONS, size=n, p=w / w.sum())
    intl = np.isin(dest, list(INTERNATIONAL))
    ac = np.where(intl, rng.choice(["A330", "A350"], size=n), ac)
    meals = np.array([AIRCAFT[a][1] for a in ac]) * rng.normal(1.0, 0.88, n)
    dep_minutes = rng.integers(5 * 60, 23 * 60, size=n)
    delay = np.where(rng.random(n) < 0.04, rng.gamma(2.0, 6.0, n), 0).round().astype(int)
    df = pd.DataFrame({
        "flight_date": pd.to_datetime(flight_date).strftime("%Y-%m-%d"),
        "flight_no": ["DL" + str(x) for x in rng.integers(100, 2999, n)],
        "origin": "ATL",
        "destination": dest,
        "aircraft_type": ac,
        "scheduled_departure": [f"{m // 60:02d}:{m % 60:02d}" for m in dep_minutes],
        "meals_loaded": meals.round().astype(int),
        "catering_delay_min": delay,
    })
    df.insert(0, "flight_id", np.arange(1, n + 1))
    return df

# The main program
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-07-01")
    ap.add_argument("--end", default="2026-09-30")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    dates = pd.date_range(args.start, args.end, freq = "D")

    counts = daily_flight_counts(dates, rng)
    hpf = hours_per_flight(dates, rng)
    flights = build_flights(dates, counts, rng)

    DEFAULT_DB.parent.mkdir(parents = True, exist_ok=True)
    with sqlite3.connect(DEFAULT_DB) as con:
        flights.to_sql("flights", con, index=False, if_exists="replace")

    print(f"Database written to {DEFAULT_DB}")
    print(f"   flights: {len(flights):,} rows")

if __name__ == "__main__":
    main()





