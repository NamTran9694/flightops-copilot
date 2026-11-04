"""Tests for the AI tools. Run with: pytest -v"""
import json

from flightops.tools import TOOL_SCHEMAS, TOOLS, execute_tool


def test_every_schema_has_a_function():
    names = [t["function"]["name"] for t in TOOL_SCHEMAS]
    assert sorted(names) == sorted(TOOLS)


def test_run_sql_tool_returns_rows():
    out, is_error = execute_tool("run_sql", {"query": "SELECT COUNT(*) AS n FROM flights"})
    assert not is_error
    assert json.loads(out)["rows"][0][0] == 623585


def test_search_tool_finds_policy():
    out, is_error = execute_tool("search_policies", {"query": "weekly safety bonus", "k": 1})
    assert not is_error and "40 dollar" in out


def test_errors_are_returned_not_raised():
    out, is_error = execute_tool("run_sql", {"query": "DROP TABLE flights"})
    assert is_error and "SQLGuardError" in json.loads(out)["error"]
    out, is_error = execute_tool("unknown_tool", {})
    assert is_error
    out, is_error = execute_tool("forecast_metric", {"metric": "revenue"})
    assert is_error and "ValueError" in out