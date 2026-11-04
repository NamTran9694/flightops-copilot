"""REST API so other systems (Power BI, Power Automate, a Teams bot) can ask the copilot.

Run:   uvicorn app.api:app --reload
Docs:  http://127.0.0.1:8000/docs
"""
import sys
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flightops.agent import FlightOpsAgent  # noqa: E402

app = FastAPI(title="FlightOps Copilot API", version="1.0.0")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)


class StepOut(BaseModel):
    tool: str
    input: dict
    is_error: bool


class AskResponse(BaseModel):
    answer: str
    steps: list[StepOut]
    seconds: float
    tokens: int


@lru_cache(maxsize=1)
def get_agent() -> FlightOpsAgent:
    return FlightOpsAgent()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, agent: FlightOpsAgent = Depends(get_agent)) -> AskResponse:
    try:
        res = agent.ask(req.question)
    except Exception as e:  # noqa: BLE001 - return a clean error instead of a crash
        raise HTTPException(status_code=503, detail=f"AI service error: {type(e).__name__}") from e
    return AskResponse(
        answer=res.answer,
        steps=[StepOut(tool=s.tool, input=s.input, is_error=s.is_error) for s in res.steps],
        seconds=res.seconds,
        tokens=res.input_tokens + res.output_tokens,
    )