# ------ A. Setup and secrets
"""Chat page: ask the FlightOps Copilot (AI agent) questions in plain English."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

st.set_page_config(page_title="Ask the Copilot", page_icon="💬", layout="wide")

# ---------- A. Secrets: on the cloud the API key lives in Streamlit "Secrets", locally in .env ----------
try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ.setdefault("GROQ_API_KEY", st.secrets["GROQ_API_KEY"])
except Exception:  # no secrets file on your computer -> .env is used
    pass

from flightops.agent import FlightOpsAgent  # noqa: E402
from flightops.config import settings  # noqa: E402

MAX_QUESTIONS = 10          # per visitor session, protects the free API quota
EXAMPLES = [
    "What was our hours per flight last month? Are we on target?",
    "Did driver attrition change after the March 2025 incentive program?",
    "How many FTEs and trucks do we need for December 2026?",
    "What must a driver do before raising the highloader?",
]

# ---------- B. Database + agent (created once, shared by all visitors) ----------
@st.cache_resource(show_spinner="Building the synthetic database (first start only)...")
def ensure_database() -> bool:
    if not settings.db_path.exists() or settings.db_path.stat().st_size < 1_000_000:
        subprocess.run([sys.executable, str(ROOT / "scripts" / "generate_data.py")], check=True)
    return True


@st.cache_resource
def get_agent() -> FlightOpsAgent:
    return FlightOpsAgent()


ensure_database()
try:
    agent = get_agent()
except RuntimeError as e:   # missing API key
    st.error(f"The AI is not configured: {e}")
    st.stop()

# ---------- C. Show the audit trail: every tool the AI used ----------
def show_steps(steps: list[dict]) -> None:
    if not steps:
        return
    with st.expander(f"🔍 How I got this ({len(steps)} tool call{'s' if len(steps) > 1 else ''})"):
        for i, step in enumerate(steps, 1):
            badge = "❌ error (the AI fixed it)" if step["is_error"] else "✅"
            st.markdown(f"**{i}. `{step['tool']}`** {badge}")
            if step["tool"] == "run_sql":
                st.code(step["input"].get("query", ""), language="sql", wrap_lines=True)
            else:
                st.json(step["input"])
            try:
                data = json.loads(step["output"])
            except json.JSONDecodeError:
                data = {"raw": step["output"][:500]}
            if "error" in data:
                st.error(data["error"])
            elif "rows" in data:
                st.dataframe(pd.DataFrame(data["rows"], columns=data["columns"]).head(20), hide_index=True)
            elif "results" in data:
                st.caption("Policy sections found: " + ", ".join(r["source"] for r in data["results"]))
            else:
                st.json(data, expanded=False)

# ---------- D. Page layout ----------
st.title("💬 Ask the FlightOps Copilot")
st.caption("An AI agent that queries the operations database, searches company policies and runs forecasts. "
           "All data is synthetic.")

if "messages" not in st.session_state:
    st.session_state.messages = []      # each item: {"role", "content", "steps", "meta"}
    st.session_state.asked = 0

with st.sidebar:
    st.markdown(f"**Model:** `{settings.model}`")
    st.markdown(f"**Questions left:** {MAX_QUESTIONS - st.session_state.asked} / {MAX_QUESTIONS}")
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

for m in st.session_state.messages:     # re-draw the conversation (Streamlit reruns the script each time)
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        show_steps(m.get("steps", []))
        if m.get("meta"):
            st.caption(m["meta"])

question = st.chat_input("Ask about productivity, safety, attrition, forecasts...")
if not st.session_state.messages and not question:
    examples_box = st.empty()               # a placeholder we can clear later
    with examples_box.container():
        st.markdown("**Try an example:**")
        cols = st.columns(len(EXAMPLES))
        for col, ex in zip(cols, EXAMPLES):
            if col.button(ex, width="stretch"):
                question = ex
    if question:
        examples_box.empty()                # hide the examples once a question is asked

# ---------- E. Answer a new question ----------
if question:
    if st.session_state.asked >= MAX_QUESTIONS:
        st.warning("You reached the question limit for this demo. Thank you for trying it!")
        st.stop()
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[-6:]]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking, querying data and reading policies..."):
            try:
                res = agent.ask(question, history=history)
            except Exception as e:  # noqa: BLE001 - show a friendly message, not a crash
                st.error("The free AI service is busy or unavailable. Please wait a minute and try again.")
                st.caption(f"Details: {type(e).__name__}")
                st.session_state.messages.pop()
                st.stop()
        steps = [s.__dict__ for s in res.steps]
        meta = f"{res.seconds}s · {len(steps)} tool calls · {res.input_tokens + res.output_tokens:,} tokens"
        st.markdown(res.answer)
        show_steps(steps)
        st.caption(meta)

    st.session_state.asked += 1
    st.session_state.messages.append({"role": "assistant", "content": res.answer, "steps": steps, "meta": meta})
    st.rerun()                              # redraw so the sidebar counter updates