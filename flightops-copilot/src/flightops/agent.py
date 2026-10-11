"""The agent loop: the LLM decides which tool to call, we run it, and repeat until it answers."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from .config import settings
from .db import describe_schema
from .prompts import SYSTEM_PROMPT
from .tools import TOOL_SCHEMAS, execute_tool


@dataclass
class Step:
    tool: str
    input: dict
    output: str
    is_error: bool
    seconds: float


@dataclass
class AgentResult:
    answer: str
    steps: list[Step] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    stopped_early: bool = False


class FlightOpsAgent:
    def __init__(self, client: Any | None = None, model: str | None = None):
        if client is None:
            import anthropic  # imported here so tests can run without an API key
            client = anthropic.Anthropic()
        self.client = client
        self.model = model or settings.model
        self.system = SYSTEM_PROMPT.format(schema=describe_schema())

    def ask(self, question: str, history: list[dict] | None = None) -> AgentResult:
        """Answer one question. `history` is a list of earlier {'role','content'} text messages."""
        t0 = time.time()
        messages: list[dict] = list(history or []) + [{"role": "user", "content": question}]
        result = AgentResult(answer="")

        for _ in range(settings.max_agent_steps):
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=settings.max_tokens,
                # cache the long, unchanging system prompt to cut cost and latency
                system=[{"type": "text", "text": self.system, "cache_control": {"type": "ephemeral"}}],
                tools=TOOL_SCHEMAS,
                messages=messages,
            )
            result.input_tokens += getattr(resp.usage, "input_tokens", 0) or 0
            result.output_tokens += getattr(resp.usage, "output_tokens", 0) or 0
            messages.append({"role": "assistant", "content": resp.content})

            if resp.stop_reason != "tool_use":
                result.answer = "".join(b.text for b in resp.content if b.type == "text").strip()
                break

            tool_results = []
            for block in resp.content:
                if block.type != "tool_use":
                    continue
                s = time.time()
                output, is_error = execute_tool(block.name, dict(block.input))
                result.steps.append(Step(block.name, dict(block.input), output, is_error, round(time.time() - s, 3)))
                tool_results.append({"type": "tool_result", "tool_use_id": block.id,
                                     "content": output[:20_000], "is_error": is_error})
            messages.append({"role": "user", "content": tool_results})
        else:
            result.stopped_early = True
            result.answer = ("I could not finish within the step limit. "
                             "Please ask a more specific question.")

        result.seconds = round(time.time() - t0, 2)
        return result


def result_to_dict(r: AgentResult) -> dict:
    return {
        "answer": r.answer,
        "steps": [s.__dict__ for s in r.steps],
        "input_tokens": r.input_tokens,
        "output_tokens": r.output_tokens,
        "seconds": r.seconds,
        "stopped_early": r.stopped_early,
    }


if __name__ == "__main__":  # quick CLI: python -m flightops.agent "your question"
    import sys
    q = " ".join(sys.argv[1:]) or "What was the average hours per flight last month?"
    out = FlightOpsAgent().ask(q)
    for st in out.steps:
        print(f"[tool] {st.tool} {json.dumps(st.input)[:200]}")
    print("\n" + out.answer)
    print(f"\n({out.seconds}s, {out.input_tokens} in / {out.output_tokens} out tokens)")
