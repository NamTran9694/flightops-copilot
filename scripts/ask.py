"""Ask the agent one question: python scripts/ask.py "your question" """
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flightops.agent import FlightOpsAgent  # noqa: E402
from flightops.config import settings  # noqa: E402

question = " ".join(sys.argv[1:]) or "What was the average hours per flight last month? Are we on target?"
print(f"Model: {settings.provider} / {settings.model}")
print("QUESTION:", question, "\n")

result = FlightOpsAgent().ask(question)

for i, step in enumerate(result.steps, 1):
    status = "ERROR" if step.is_error else "ok"
    print(f"Step {i}: {step.tool} [{status}] {json.dumps(step.input)[:150]}")
print("\nANSWER:\n" + result.answer)
print(f"\n({result.seconds}s, {len(result.steps)} tool calls, "
      f"{result.input_tokens} input + {result.output_tokens} output tokens)")