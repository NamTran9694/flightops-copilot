"""Your first AI call: python scripts/hello_llm.py"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flightops.config import make_client, settings   # noqa: E402

client = make_client()
print(f"Provider: {settings.provider} | Model: {settings.model}\n")

messages = [
    {"role": "system", "content": "You are a concise operations analyst for an airline catering company."
                                    "Answer in 2-3 short sentences"},
    {"role": "user", "content": "What is our facility's hours-per-flight target, and what was our driver attrition last month?"},
]

start = time.time()
response = client.chat.completions.create(
    model = settings.model,
    messages = messages,
    temperature=0,
    max_tokens=settings.max_tokens,
)

seconds = time.time() - start

answer = response.choices[0].message.content
usage = response.usage
print("AI answer:\n" + answer)
print(f"\n({seconds:.1f} s | {usage.prompt_tokens} input tokens + {usage.completion_tokens} output tokens)")