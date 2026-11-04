"""Tools the AI can call. Each tool = a Python function + a description (schema) for the AI."""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Callable

from . import db, forecast
from .rag import DocIndex, default_embedder

# ----------- A. The python functions behind each tool ---------------------
@lru_cache(maxsize=1)
def get_index() -> DocIndex:
    """Build the policy search index once and reuse it."""
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

# ---------- B. Tool descriptions for the AI (JSON schema) ----------
def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"type": "function",
            "function": {"name": name, "description": description,
                         "parameters": {"type": "object", "properties": properties, "required": required}}}

TOOL_SCHEMAS: list[dict[str, Any]] = [
    _tool("run_sql",
          "Run ONE read-only SQLite SELECT query on the operations database and return the rows. "
          "Use it for any number, count, trend or comparison from the data. Aggregate in SQL "
          "(GROUP BY, AVG, SUM) instead of returning raw rows. Max 200 rows are returned.",
          {"query": {"type": "string", "description": "A single SQLite SELECT statement."}},
          ["query"]),
    _tool("search_policies",
          "Search company policies, SOPs, KPI definitions and playbooks. Use it for rules, targets, "
          "formulas, procedures and the reasons behind programs (for example the driver incentive).",
          {"query": {"type": "string"},
           "k": {"type": "integer", "description": "Number of sections to return (1-8).", "default": 4}},
          ["query"]),
    _tool("forecast_metric",
          "Forecast a daily metric for the next N days with a trend + seasonality model. "
          "Returns the forecast, an approximate range and the back-test error (MAPE).",
          {"metric": {"type": "string", "enum": sorted(forecast.METRICS)},
           "horizon_days": {"type": "integer", "minimum": 1, "maximum": 365}},
          ["metric"]),
    _tool("plan_resources",
          "Estimate labor hours, FTE headcount and catering trucks needed for a FUTURE date range, "
          "following the facility budget planning rules.",
          {"start_date": {"type": "string", "description": "YYYY-MM-DD, after the last data date."},
           "end_date": {"type": "string", "description": "YYYY-MM-DD"}},
          ["start_date", "end_date"]),
]

# ---------- C. Run a tool safely ----------
def execute_tool(name: str, args: dict) -> tuple[str, bool]:
    """Run a tool. Returns (json_text, is_error). Errors go back to the AI so it can fix itself."""
    fn = TOOLS.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool {name}"}), True
    try:
        return json.dumps(fn(**args), default=str), False
    except Exception as e:  # noqa: BLE001 - every error is reported to the AI
        return json.dumps({"error": f"{type(e).__name__}: {e}"}), True