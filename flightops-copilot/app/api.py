"""REST API so other systems (Power BI, Teams bot, Power Automate) can call the agent.

Run:  uvicorn app.api:app --reload
Then: POST http://127.0.0.1:8000/ask  {"question": "..."}
"""
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flightops.agent import FlightOpsAgent, result_to_dict  # noqa: E402

app = FastAPI(title="FlightOps Copilot API", version="1.0.0")
_agent: FlightOpsAgent | None = None


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(req: AskRequest):
    global _agent
    try:
        _agent = _agent or FlightOpsAgent()
        return result_to_dict(_agent.ask(req.question))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e)) from e
