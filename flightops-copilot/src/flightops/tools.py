"""Tools the agent can call. Each tool has a JSON schema (for the LLM) and a Python function."""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Callable

from . import db, forecast
from .rag import DocIndex, default_embedder


@lru_cache(maxsize=1)
def get_index() -> DocIndex:
    return DocIndex.from_dir(embedder=default_embedder())


def tool_run_sql(query: str) -> dict:
    return db.run_query(query)


def tool_search_policies(query: str, k: int = 4) -> dict:
    return {"results": get_index().search(query, k=int(k))}


def tool_forecast_metric(metric: str, horizon_days: int = 30) -> dict:
    return forecast.forecast_metric(metric, int(horizon_days))


def tool_plan_resources(start_date: str, end_date: str) -> dict:
    return forecast.plan_resources(start_date, end_date)


TOOLS: dict[str, Callable[..., dict]] = {
    "run_sql": tool_run_sql,
    "search_policies": tool_search_policies,
    "forecast_metric": tool_forecast_metric,
    "plan_resources": tool_plan_resources,
}

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "run_sql",
        "description": (
            "Run ONE read-only SQLite SELECT query on the operations database and return the rows. "
            "Use it for any number, count, trend or comparison from the data. Aggregate in SQL "
            "(GROUP BY, AVG, SUM) instead of returning raw rows. Max 200 rows are returned."),
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "A single SQLite SELECT statement."}},
            "required": ["query"],
        },
    },
    {
        "name": "search_policies",
        "description": (
            "Search company policies, SOPs, KPI definitions and playbooks. Use it for rules, targets, "
            "formulas, procedures and the reasons behind programs (for example the driver incentive)."),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "k": {"type": "integer", "description": "Number of sections to return (1-8).", "default": 4},
            },
            "required": ["query"],
        },
    },
    {
        "name": "forecast_metric",
        "description": (
            "Forecast a daily metric for the next N days with a trend + seasonality model. "
            "Returns the forecast, an approximate range and the back-test error (MAPE)."),
        "input_schema": {
            "type": "object",
            "properties": {
                "metric": {"type": "string", "enum": sorted(forecast.METRICS)},
                "horizon_days": {"type": "integer", "minimum": 1, "maximum": 365},
            },
            "required": ["metric"],
        },
    },
    {
        "name": "plan_resources",
        "description": (
            "Estimate labor hours, FTE headcount and catering trucks needed for a FUTURE date range, "
            "following the facility budget planning rules."),
        "input_schema": {
            "type": "object",
            "properties": {
                "start_date": {"type": "string", "description": "YYYY-MM-DD, after the last data date."},
                "end_date": {"type": "string", "description": "YYYY-MM-DD"},
            },
            "required": ["start_date", "end_date"],
        },
    },
]


def execute_tool(name: str, args: dict) -> tuple[str, bool]:
    """Run a tool. Returns (json_text, is_error). Errors go back to the model so it can fix itself."""
    fn = TOOLS.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool {name}"}), True
    try:
        return json.dumps(fn(**args), default=str), False
    except Exception as e:  # noqa: BLE001 - every error is reported to the model
        return json.dumps({"error": f"{type(e).__name__}: {e}"}), True
