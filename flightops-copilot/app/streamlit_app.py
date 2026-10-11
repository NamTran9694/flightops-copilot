"""Streamlit UI: chat with the agent + a KPI overview tab.

Run:  streamlit run app/streamlit_app.py
"""
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flightops.agent import FlightOpsAgent  # noqa: E402
from flightops.db import run_query  # noqa: E402

st.set_page_config(page_title="FlightOps Copilot", page_icon="✈️", layout="wide")

EXAMPLES = [
    "What was our average labor hours per flight in each of the last 6 months? Are we meeting the target?",
    "Did driver attrition change after the March 2025 incentive program? Show monthly numbers before and after.",
    "How many recordable safety incidents did we have per quarter, and what is the target?",
    "Forecast daily flight volume for the next 30 days.",
    "How many FTEs and trucks do we need for December 2026?",
    "What must a driver do before raising the highloader platform?",
]


@st.cache_resource
def get_agent():
    return FlightOpsAgent()


@st.cache_data
def monthly_kpis():
    q = """SELECT substr(ops_date,1,7) AS month, ROUND(AVG(flights)) AS flights_per_day,
                  ROUND(SUM(labor_hours)/SUM(flights),2) AS hours_per_flight
           FROM daily_ops GROUP BY month ORDER BY month"""
    r = run_query(q)
    return pd.DataFrame(r["rows"], columns=r["columns"])


@st.cache_data
def monthly_incidents():
    q = """SELECT substr(incident_date,1,7) AS month, severity, COUNT(*) AS incidents
           FROM safety_incidents GROUP BY month, severity ORDER BY month"""
    r = run_query(q)
    return pd.DataFrame(r["rows"], columns=r["columns"])


st.title("✈️ FlightOps Copilot")
st.caption("AI analyst for airline catering operations · RAG + text-to-SQL + forecasting · synthetic data")

tab_chat, tab_kpi = st.tabs(["💬 Ask the copilot", "📊 KPI overview"])

with tab_kpi:
    k = monthly_kpis()
    c1, c2, c3 = st.columns(3)
    c1.metric("Hours per flight (last month)", k.hours_per_flight.iloc[-1],
              f"{k.hours_per_flight.iloc[-1] - k.hours_per_flight.iloc[0]:+.2f} vs first month",
              delta_color="inverse")
    c2.metric("Flights per day (last month)", int(k.flights_per_day.iloc[-1]))
    c3.metric("Productivity gain", f"{(1 - k.hours_per_flight.iloc[-1] / k.hours_per_flight.iloc[0]) * 100:.0f}%")
    st.plotly_chart(px.line(k, x="month", y="hours_per_flight", markers=True,
                            title="Labor hours per flight (lower is better)"), width="stretch")
    st.plotly_chart(px.bar(k, x="month", y="flights_per_day", title="Average flights per day"),
                    width="stretch")
    st.plotly_chart(px.bar(monthly_incidents(), x="month", y="incidents", color="severity",
                           title="Safety events by month"), width="stretch")

with tab_chat:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.subheader("Try a question")
        for ex in EXAMPLES:
            if st.button(ex, width="stretch"):
                st.session_state.pending = ex
        if st.button("Clear chat"):
            st.session_state.messages = []

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    prompt = st.chat_input("Ask about productivity, safety, attrition, forecasts…")
    prompt = prompt or st.session_state.pop("pending", None)
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking and querying data…"):
                history = st.session_state.messages[:-1][-6:]   # keep a short memory
                res = get_agent().ask(prompt, history=history)
            st.markdown(res.answer)
            with st.expander(f"🔎 How I got this ({len(res.steps)} tool calls, {res.seconds}s, "
                             f"{res.input_tokens + res.output_tokens:,} tokens)"):
                for i, s in enumerate(res.steps, 1):
                    st.markdown(f"**{i}. `{s.tool}`** {'❌ error' if s.is_error else ''} · {s.seconds}s")
                    if s.tool == "run_sql":
                        st.code(s.input.get("query", ""), language="sql")
                    else:
                        st.json(s.input)
                    try:
                        st.json(json.loads(s.output), expanded=False)
                    except json.JSONDecodeError:
                        st.text(s.output[:2000])
        st.session_state.messages.append({"role": "assistant", "content": res.answer})
