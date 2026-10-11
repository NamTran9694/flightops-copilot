"""Unit tests. They need the synthetic database (python scripts/generate_data.py) but no API key."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from flightops import db, forecast
from flightops.agent import FlightOpsAgent
from flightops.rag import BM25, DocIndex, chunk_markdown, rrf, tokenize
from flightops.tools import execute_tool

pytestmark = pytest.mark.skipif(not db.settings.db_path.exists(),
                                reason="Run scripts/generate_data.py first")


# ---------- SQL guardrails ----------
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


def test_sql_allows_select_and_cte():
    r = db.run_query("WITH x AS (SELECT COUNT(*) n FROM flights) SELECT n FROM x")
    assert r["rows"][0][0] > 1000


def test_sql_row_limit():
    r = db.run_query("SELECT flight_id FROM flights", row_limit=10)
    assert r["row_count"] == 10 and r["truncated"]


def test_connection_is_read_only():
    con = db.connect()
    with pytest.raises(Exception):
        con.execute("CREATE TABLE hack (x INT)")
    con.close()


# ---------- RAG ----------
def test_chunking_by_heading():
    chunks = chunk_markdown("doc", "# Title\n\n## A\ntext a\n\n## B\ntext b")
    assert [c.section for c in chunks] == ["A", "B"]
    assert chunks[0].source == "doc#A"


def test_bm25_ranks_relevant_doc_first():
    docs = [tokenize("truck speed limit ramp"), tokenize("meal cart lifting injury")]
    s = BM25(docs).scores(tokenize("speed limit"))
    assert s[0] > s[1]


def test_rrf_combines_rankings():
    assert rrf([[0, 1, 2], [1, 0, 2]])[2] == 2


def test_hybrid_search_with_fake_embedder():
    def fake_embed(texts):
        return np.array([[t.lower().count("speed") + 0.1, t.lower().count("bonus") + 0.1] for t in texts])
    idx = DocIndex.from_dir(embedder=fake_embed)
    top = idx.search("maximum speed near aircraft", k=1)[0]
    assert top["source"].startswith("driver_safety_policy")


# ---------- Forecasting ----------
def test_forecast_backtest_error_is_reasonable():
    f = forecast.forecast_metric("flights", 30)
    assert f["backtest_mape_pct"] < 8
    assert len(f["weekly_avg_forecast"]) >= 4


def test_forecast_rejects_unknown_metric():
    with pytest.raises(ValueError):
        forecast.forecast_metric("revenue", 30)


def test_plan_resources_future_only():
    with pytest.raises(ValueError):
        forecast.plan_resources("2020-01-01", "2020-01-31")


def test_plan_resources_values():
    p = forecast.plan_resources("2026-11-01", "2026-11-30")
    assert p["hours_per_flight_used"] >= p["hours_per_flight_floor_90d"] - 1e-6
    assert p["required_fte_incl_12pct_buffer"] > 0 and p["trucks_needed_peak_day"] > 0


# ---------- Tools ----------
def test_tool_errors_are_returned_not_raised():
    out, is_error = execute_tool("run_sql", {"query": "DROP TABLE flights"})
    assert is_error and "error" in json.loads(out)
    out, is_error = execute_tool("nope", {})
    assert is_error


# ---------- Agent loop with a fake LLM ----------
class FakeClient:
    """Mimics anthropic.Anthropic().messages.create: first asks for a tool, then answers."""

    def __init__(self):
        self.calls = 0
        self.messages = self

    def create(self, **kwargs):
        self.calls += 1
        usage = SimpleNamespace(input_tokens=100, output_tokens=20)
        if self.calls == 1:
            block = SimpleNamespace(type="tool_use", id="t1", name="run_sql",
                                    input={"query": "SELECT COUNT(*) FROM safety_incidents"})
            return SimpleNamespace(stop_reason="tool_use", content=[block], usage=usage)
        last = kwargs["messages"][-1]["content"][0]
        assert last["type"] == "tool_result" and not last["is_error"]
        n = json.loads(last["content"])["rows"][0][0]
        text = SimpleNamespace(type="text", text=f"There were {n} incidents.")
        return SimpleNamespace(stop_reason="end_turn", content=[text], usage=usage)


def test_agent_loop_calls_tool_then_answers():
    res = FlightOpsAgent(client=FakeClient(), model="fake").ask("How many incidents?")
    assert res.steps[0].tool == "run_sql"
    assert res.answer.startswith("There were")
    assert res.input_tokens == 200
