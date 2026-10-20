"""Generate a realistic SYNTHETIC dataset for an airline catering facility."""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = path(__file__).resolve().parents[1]
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


