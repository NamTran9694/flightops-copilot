"""Tests for the safe SQL layer. Run with: pytest -v"""
import pytest

from flightops import db


# ---------- 1. Unsafe queries must be blocked ----------
@pytest.mark.parametrize("bad", [
    "DROP TABLE flights",
    "DELETE FROM flights",
    "SELECT 1; DROP TABLE flights",
    "PRAGMA table_info(flights)",
    "ATTACH DATABASE 'x.db' AS x",
    "",
])
def test_sql_guard_blocks_unsafe(bad):
    with pytest.raises(db.SQLGuardError):
        db.validate_sql(bad)


# ---------- 2. Normal SELECT and WITH queries must work ----------
def test_sql_allows_select_and_cte():
    r = db.run_query("WITH x AS (SELECT COUNT(*) AS n FROM flights) SELECT n FROM x")
    assert r["rows"][0][0] == 623585


# ---------- 3. Row limit ----------
def test_sql_row_limit():
    r = db.run_query("SELECT flight_id FROM flights", row_limit=10)
    assert r["row_count"] == 10
    assert r["truncated"] is True


# ---------- 4. The connection itself is read-only ----------
def test_connection_is_read_only():
    con = db.connect()
    with pytest.raises(Exception):
        con.execute("CREATE TABLE hack (x INT)")
    con.close()


# ---------- 5. Schema description for the AI ----------
def test_describe_schema_lists_tables():
    text = db.describe_schema()
    for name in ["flights", "labor_daily", "employees", "safety_incidents", "safety_audits", "daily_ops"]:
        assert name in text