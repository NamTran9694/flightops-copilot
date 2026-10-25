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
AIRCRAFT = { #aircaft type -> (share of flights, average meals)
"A321": (0.33, 160), "B737-900": (0.28, 150), "A220": (0.17, 90),
"B757": (0.14, 170), "A330": (0.05, 260), "A350": (0.03, 300),
}

DEPARTMENTS = {  # share of daily labor hours
    "Drivers": 0.30, "Loaders": 0.22, "Kitchen": 0.25, "Warehouse": 0.15, "Supervisors": 0.08,
}

POLICY_DATE = pd.Timestamp("2025-03-01")   # driver +$2/h raise + safety incentive
AUDIT_START = pd.Timestamp("2025-01-06")   # weekly driver safety audit begin

# How many flights each day?
def daily_flight_counts(dates: pd.DatetimeIndex, rng:np.random.Generator) -> np.ndarray:
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
    types = list(AIRCRAFT)
    ac = rng.choice(types, size=n, p=[AIRCRAFT[a][0] for a in types])
    w = np.array([0.35 if d in INTERNATIONAL else 1.0 for d in DESTINATIONS])
    dest = rng.choice(DESTINATIONS, size=n, p=w / w.sum())
    intl = np.isin(dest, list(INTERNATIONAL))
    ac = np.where(intl, rng.choice(["A330", "A350"], size=n), ac)
    meals = np.array([AIRCRAFT[a][1] for a in ac]) * rng.normal(1.0, 0.88, n)
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

def build_labor(dates, counts, hpf, rng) -> pd.DataFrame:
    """Daily paid hours per department = flights x hours-per-flight x department share."""
    rows = []
    total = counts * hpf
    for dept, share in DEPARTMENTS.items():
        hours = total * share * rng.normal(1.0, 0.02, len(dates))
        ot = hours * np.clip(rng.normal(0.08, 0.03, len(dates)), 0.0, 0.2)
        rows.append(pd.DataFrame({
            "work_date": dates.strftime("%Y-%m-%d"),
            "department": dept,
            "headcount": np.round(hours / 8.4).astype(int),
            "regular_hours": (hours - ot).round(1),
            "overtime_hours": ot.round(1),
        }))
    return pd.concat(rows, ignore_index=True)

# The employee Simulation
def build_employees(start, end, rng) -> pd.DataFrame:
    """Simulate hiring and termination per role, month by month."""
    # --- A. Settitngs: role -> (starting headcount, base hourly rate in $)
    roles = {"Driver": (470,18.0), "Loader": (400, 17.0), "Kitchen": (590, 16.5), "Warehouse": (360, 16.0), "Supervisor": (125, 25.0)}
    months = pd.date_range(start, end, freq="MS")
    growth = np.linspace(1.0, 1.35, len(months))
    emp, eid = [], 1

    for role, (base_n, rate) in roles.items():
        active: list[dict] = []
        # --- B. Staff who already worked here before the start date
        for _ in range(base_n):
            hire = start -pd.Timedelta(days=int(rng.integers(30, 1500)))
            active.append({"employee_id": eid, "role": role, "hire_date": hire, "term_date": pd.NaT,
                           "hourly_rate": round(rate + rng.normal(0, 0.8), 2)})
            eid += 1
        for i, m in enumerate(months):
            # --- C. Attrition Rate for this month
            if role == "Driver":
                monthly_attr = 0.055 if m < POLICY_DATE else 0.015
            else:
                monthly_attr = 0.03
            # --- D. Terminations: each active person may leave this month
            for e in active:
                if pd.isna(e["term_date"]) and rng.random() < monthly_attr:
                    e["term_date"] = m + pd.Timedelta(days=int(rng.integers(0, 27)))
            # --- E. Hiring: fill the gap up to the target headcount
            current = sum(pd.isna(e["term_date"]) or e["term_date"] > m for e in active)
            target = int(base_n * growth[i])
            for _ in range(max(0, target - current)):
                r = rate + (2.0 if role == "Driver" and m >= POLICY_DATE else 0.0)
                active.append({"employee_id": eid, "role": role,
                               "hire_date": m + pd.Timedelta(days=int(rng.integers(0, 27))),
                               "term_date": pd.NaT, "hourly_rate": round(r + rng.normal(0, 0.8), 2)})
                eid += 1

        # existing drivers also get the +$2 raise on the policy date
        if role == "Driver":
            for e in active:
                if pd.isna(e["term_date"]) or e["term_date"] >= POLICY_DATE:
                    if e["hire_date"] < POLICY_DATE:
                        e["hourly_rate"] = round(e["hourly_rate"] + 2.0, 2)
        emp.extend(active)

    # --- F. Clean up and format date as text
    df = pd.DataFrame(emp)
    df = df[df["hire_date"] <= end].copy()
    df.loc[df["term_date"] > end, "term_date"] = pd.NaT
    df["hire_date"] = df["hire_date"].dt.strftime("%Y-%m-%d")
    df["term_date"] = df["term_date"].dt.strftime("%Y-%m-%d")

    return df

# The safety function
def build_safety(dates, employees, rng):
    """Daily safety incidents and weekly driver audits."""
    # --- A. Settings: incident types (with share) and descriptions
    drivers = employees[employees.role == "Driver"]
    incidents = []
    types = {"vehicle_collision": 0.3, "aircraft_contact": 0.15, "slip_trip_fall": 0.25,
             "lifting_injury": 0.20, "highloader_malfunction": 0.10}
    desc = {
        "vehicle_collision": "Catering truck made contact with ground equipment on the ramp.",
        "aircraft_contact": "Highloader platform touched aircraft door frame during docking.",
        "slip_trip_fall": "Employee slipped on wet surface in the loading area.",
        "lifting_injury": "Strain reported while lifting a full meal cart.",
        "highloader_malfunction": "Hydraulic lift stopped working during service.",       
    }
    # --- B. Incidents: each day, each severity has a small chance to happen
    for d in dates:
        # recordable incidents: ~1 / month before audits, ~1 per 2.5 months after
        p_rec = 1 / 20 if d < AUDIT_START else 1 / 75
        # near-misses / first-aid are more frequent and also trend down
        p_minor = 0.25 if d < AUDIT_START else 0.12
        for sev, p in (("recordable", p_rec), ("first_aid", p_minor * 0.4), ("near_miss", p_minor)):
            if rng.random() < p:
                t = rng.choice(list(types), p=list(types.values()))
                active = drivers[(drivers.hire_date <= d.strftime("%Y-%m-%d")) &
                                 (drivers.term_date.isna() | (drivers.term_date >= d.strftime("%Y-%m-%d")))]
                emp_id = int(active.sample(1, random_state=int(rng.integers(1e9))).employee_id.iloc[0]) if len(active) else None
                incidents.append({"incident_date": d.strftime("%Y-%m-%d"), "employee_id": emp_id,
                                  "incident_type": t, "severity": sev, "description": desc[t]})
    inc = pd.DataFrame(incidents)
    inc.insert(0, "incident_id", np.arange(1, len(inc) + 1))

    # --- c. Weekly audits: every Monday, audit about 35% of active drivers
    audits = []
    weeks = pd.date_range(max(AUDIT_START, dates[0]), dates[-1], freq="W-MON")
    for w in weeks:
        ws = w.strftime("%Y-%m-%d")
        active = drivers[(drivers.hire_date <= ws) & (drivers.term_date.isna() | (drivers.term_date >= ws))]
        sample = active.sample(frac=0.35, random_state=int(rng.integers(1e9)))
        progress = (w - AUDIT_START).days / 700
        scores = np.clip(rng.normal(82 + 10 * min(progress, 1), 7, len(sample)), 40, 100).round(1)
        audits.append(pd.DataFrame({"audit_date": ws, "employee_id": sample.employee_id.values,
                                    "score": scores, "passed": (scores >= 80).astype(int)}))

    aud = pd.concat(audits, ignore_index=True) if audits else pd.DataFrame(
        columns=["audit_date", "employee_id",  "score", "passed"])
    aud.insert(0, "audit_id", np.arange(1, len(aud) + 1))
    return inc, aud

# The SQL View:
SCHEMA_SQL = """
CREATE INDEX IF NOT EXISTS idx_flights_date ON flights(flight_date);
CREATE INDEX IF NOT EXISTS idx_labor_date ON labor_daily(work_date);
CREATE INDEX IF NOT EXISTS idx_inc_date ON safety_incidents(incident_date);
CREATE INDEX IF NOT EXISTS idx_aud_date ON safety_audits(audit_date);
DROP VIEW IF EXISTS daily_ops;
CREATE VIEW daily_ops AS
SELECT f.flight_date AS ops_date,
        f.flights,
        f.meals,
        l.labor_hours,
        ROUND(l.labor_hours * 1.0 / f.flights, 2) AS hours_per_flight
FROM (SELECT flight_date, COUNT(*) AS flights, SUM(meals_loaded) AS meals
        FROM flights GROUP BY flight_date) f
JOIN (SELECT work_date, SUM(regular_hours + overtime_hours) AS labor_hours
        FROM labor_daily GROUP BY work_date) l
    ON l.work_date = f.flight_date;
"""

# The main program
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-07-01")
    ap.add_argument("--end", default="2026-09-30")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    dates = pd.date_range(start, end, freq = "D")

    counts = daily_flight_counts(dates, rng)
    hpf = hours_per_flight(dates, rng)
    flights = build_flights(dates, counts, rng)
    labor = build_labor(dates, counts, hpf, rng)
    employees = build_employees(start, end, rng)
    incidents, audits = build_safety(dates, employees, rng)

    DEFAULT_DB.parent.mkdir(parents = True, exist_ok=True)
    if DEFAULT_DB.exists():
        DEFAULT_DB.unlink()
    with sqlite3.connect(DEFAULT_DB) as con:
        flights.to_sql("flights", con, index=False)
        labor.to_sql("labor_daily", con, index=False)
        employees.to_sql("employees", con, index=False)
        incidents.to_sql("safety_incidents", con, index=False)
        audits.to_sql("safety_audits", con, index=False)
        con.executescript(SCHEMA_SQL)

    print(f"Database written to {DEFAULT_DB}")
    print(f"   flights: {len(flights):,} rows")
    print(f"   labor_daily: {len(labor):,} rows")
    print(f"   employees: {len(employees):,} rows")
    print(f"   safety_incidents: {len(incidents):,} rows")
    print(f"   safety_audits: {len(audits):,} rows")

if __name__ == "__main__":
    main()





