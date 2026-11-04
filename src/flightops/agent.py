# -------- A. Result Container -----------
"""The agent loop: the LLM decides which tool to call, we run it, and repeat until it answers.

Works with any OpenAI-compatible provider (Ollama, Groq, Gemini ...) - see config.py.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from .config import make_client, settings
from .db import describe_schema
from .prompts import SYSTEM_PROMPT
from .tools import TOOL_SCHEMAS, execute_tool

# some open models (e.g. Qwen3) "think out loud" inside <think> tags - hide that from the user
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


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


# ---------- B. The Loop ---------------------
class FlightOpsAgent:
    def __init__(self, client: Any | None = None, model: str | None = None):
        self.client = client or make_client()
        self.model = model or settings.model
        self.system = SYSTEM_PROMPT.format(schema=describe_schema())

    def ask(self, question: str, history: list[dict] | None = None) -> AgentResult:
        """Answer one question. `history` is a list of earlier {'role','content'} text messages."""
        t0 = time.time()
        messages: list[dict] = [{"role": "system", "content": self.system}]
        messages += list(history or [])
        messages.append({"role": "user", "content": question})
        result = AgentResult(answer="")

        for _ in range(settings.max_agent_steps):
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
                temperature=0,              # same question -> same answer (good for evaluation)
                max_tokens=settings.max_tokens,
            )
            usage = getattr(resp, "usage", None)
            result.input_tokens += getattr(usage, "prompt_tokens", 0) or 0
            result.output_tokens += getattr(usage, "completion_tokens", 0) or 0
            msg = resp.choices[0].message
            tool_calls = msg.tool_calls or []

            if not tool_calls:                           # no tool requested -> final answer
                result.answer = _THINK.sub("", msg.content or "").strip()
                break

            # 1) remember what the model asked for
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [{"id": c.id, "type": "function",
                                "function": {"name": c.function.name, "arguments": c.function.arguments}}
                               for c in tool_calls],
            })
            # 2) run each tool and send the result back
            for call in tool_calls:
                s = time.time()
                try:
                    args = json.loads(call.function.arguments or "{}")
                    output, is_error = execute_tool(call.function.name, args)
                except json.JSONDecodeError:
                    args = {"raw": call.function.arguments}
                    output, is_error = json.dumps({"error": "Arguments were not valid JSON. Try again."}), True
                result.steps.append(Step(call.function.name, args, output, is_error, round(time.time() - s, 3)))
                messages.append({"role": "tool", "tool_call_id": call.id,
                                 "content": output[: settings.tool_output_chars]})
        else:
            result.stopped_early = True
            result.answer = "I could not finish within the step limit. Please ask a more specific question."

        result.seconds = round(time.time() - t0, 2)
        return result