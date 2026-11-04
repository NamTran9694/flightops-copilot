"""Test the REST API with the fake AI from test_agent.py (no API key needed)."""
from fastapi.testclient import TestClient

from app.api import app, get_agent
from flightops.agent import FlightOpsAgent
from test_agent import FakeClient

app.dependency_overrides[get_agent] = lambda: FlightOpsAgent(client=FakeClient(), model="fake")
client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_ask_returns_answer_and_steps():
    r = client.post("/ask", json={"question": "How many safety incidents?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"].startswith("There were")
    assert body["steps"][0]["tool"] == "run_sql"


def test_question_too_short_is_rejected():
    r = client.post("/ask", json={"question": "hi"})
    assert r.status_code == 422          # FastAPI validates the input for us