"""Safe, read-only SQL access for the agent (text-to-SQL guardrails)."""
from __future__ import annotations

import re
import sqlite3
import time
from pathlib import Path

from .config import settings

# Words that must never appear in an agent query (defense in depth;
# the connection itself is also opened read-only).
_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|attach|detach|pragma|vacuum|reindex|truncate)\b",
    re.IGNORECASE,
)

TABLE_NOTES = {
    "flights": "One row per catered flight departing ATL. catering_delay_min > 0 means a catering-caused delay.",
    "labor_daily": "Daily paid labor hours per department (Drivers, Loaders, Kitchen, Warehouse, Supervisors).",
    "employees": "One row per employee. term_date is NULL while the employee is still active. Dates are 'YYYY-MM-DD' text.",
    "safety_incidents": "Safety events. severity in ('near_miss','first_aid','recordable'). employee_id refers to a driver.",
    "safety_audits": "Weekly driver safety audits (started 2025-01-06). passed = 1 when score >= 80.",
    "daily_ops": "VIEW: one row per day with flights, meals, labor_hours and hours_per_flight (the main productivity KPI).",
}


class SQLGuardError(ValueError):
    """Raised when a query is not allowed."""


def validate_sql(query: str) -> str:
    """Return a cleaned query or raise SQLGuardError."""
    q = query.strip().rstrip(";").strip()
    if not q:
        raise SQLGuardError("Empty query.")
    if ";" in q:
        raise SQLGuardError("Only one SQL statement is allowed.")
    if not re.match(r"^(select|with)\b", q, re.IGNORECASE):
        raise SQLGuardError("Only SELECT (or WITH ... SELECT) queries are allowed.")
    if _FORBIDDEN.search(q):
        raise SQLGuardError("Query contains a forbidden keyword.")
    return q


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    path = Path(db_path or settings.db_path)
    if not path.exists():
        raise FileNotFoundError(f"Database not found at {path}. Run: python scripts/generate_data.py")
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


def run_query(query: str, db_path: Path | None = None, row_limit: int | None = None,
              timeout_s: float | None = None) -> dict:
    """Execute a validated read-only query. Returns columns, rows and a truncated flag."""
    q = validate_sql(query)
    limit = row_limit or settings.sql_row_limit
    timeout = timeout_s or settings.sql_timeout_s
    con = connect(db_path)
    deadline = time.monotonic() + timeout
    # abort long-running queries
    con.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
    try:
        cur = con.execute(q)
        cols = [c[0] for c in cur.description] if cur.description else []
        rows = cur.fetchmany(limit + 1)
    except sqlite3.OperationalError as e:
        if "interrupted" in str(e):
            raise SQLGuardError(f"Query took longer than {timeout} seconds and was stopped.") from e
        raise
    finally:
        con.close()
    truncated = len(rows) > limit
    return {"columns": cols, "rows": [list(r) for r in rows[:limit]],
            "row_count": min(len(rows), limit), "truncated": truncated}


def describe_schema(db_path: Path | None = None) -> str:
    """Human/LLM readable schema with column types, notes and date ranges."""
    con = connect(db_path)
    lines = []
    try:
        objs = con.execute(
            "SELECT name, type FROM sqlite_master WHERE type IN ('table','view') "
            "AND name NOT LIKE 'sqlite_%' ORDER BY type, name").fetchall()
        for name, typ in objs:
            cols = con.execute(f"PRAGMA table_info('{name}')").fetchall()
            col_txt = ", ".join(f"{c['name']} {c['type'] or 'TEXT'}" for c in cols)
            lines.append(f"- {name} ({typ}): {col_txt}")
            if name in TABLE_NOTES:
                lines.append(f"    note: {TABLE_NOTES[name]}")
        rng = con.execute("SELECT MIN(flight_date), MAX(flight_date) FROM flights").fetchone()
        lines.append(f"Data covers {rng[0]} to {rng[1]}.")
    finally:
        con.close()
    return "\n".join(lines)
