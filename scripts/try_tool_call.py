"""See which tool the AI chooses for each question: python scripts/try_tool_call.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flightops.config import make_client, settings  # noqa: E402
from flightops.db import describe_schema  # noqa: E402
from flightops.tools import TOOL_SCHEMAS, execute_tool  # noqa: E402

client = make_client()
system = ("You are an operations analyst for an airline catering facility. "
          "Use the tools to get facts; never guess numbers.\n\nDatabase:\n" + describe_schema())

questions = [
    "What is our hours-per-flight target?",
    "How many recordable safety incidents happened in 2025?",
    "Forecast average daily flights for the next 30 days.",
    "How many trucks do we need on the peak day in December 2026?",
]

for q in questions:
    print("=" * 80)
    print("QUESTION:", q)
    response = client.chat.completions.create(
        model=settings.model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": q}],
        tools=TOOL_SCHEMAS,
        tool_choice="auto",
        temperature=0,
        max_tokens=settings.max_tokens,
    )
    msg = response.choices[0].message
    if not msg.tool_calls:
        print("No tool used. AI answered directly:\n", msg.content)
        continue
    for call in msg.tool_calls:
        args = json.loads(call.function.arguments or "{}")
        print(f"AI CHOSE TOOL: {call.function.name}")
        print(f"WITH ARGUMENTS: {args}")
        output, is_error = execute_tool(call.function.name, args)
        print(("ERROR: " if is_error else "RESULT: ") + output[:300] + ("..." if len(output) > 300 else ""))