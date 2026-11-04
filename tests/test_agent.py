"""Test the agent loop with a FAKE AI, so no API key is needed."""
import json
from types import SimpleNamespace

from flightops.agent import FlightOpsAgent


def _resp(content=None, tool_calls=None):
    """Build an object that looks like an OpenAI chat response."""
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    usage = SimpleNamespace(prompt_tokens=100, completion_tokens=20)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=usage)


class FakeClient:
    """Pretends to be the AI: first it asks for run_sql, then it answers using the tool result."""

    def __init__(self, first_args='{"query": "SELECT COUNT(*) FROM safety_incidents"}'):
        self.calls = 0
        self.first_args = first_args
        self.chat = SimpleNamespace(completions=self)   # so client.chat.completions.create works

    def create(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            call = SimpleNamespace(id="t1", function=SimpleNamespace(name="run_sql", arguments=self.first_args))
            return _resp(tool_calls=[call])
        last = kwargs["messages"][-1]                   # the tool result we sent back
        assert last["role"] == "tool" and last["tool_call_id"] == "t1"
        data = json.loads(last["content"])
        if "error" in data:
            return _resp(content="<think>bad args</think>Sorry, I could not run the query.")
        return _resp(content=f"<think>counting...</think>There were {data['rows'][0][0]} incidents.")


def test_agent_loop_calls_tool_then_answers():
    res = FlightOpsAgent(client=FakeClient(), model="fake").ask("How many incidents?")
    assert res.steps[0].tool == "run_sql" and not res.steps[0].is_error
    assert res.answer.startswith("There were")      # the <think> text is removed
    assert res.input_tokens == 200                  # 2 AI calls x 100 tokens


def test_agent_handles_bad_json_arguments():
    res = FlightOpsAgent(client=FakeClient(first_args="{not json"), model="fake").ask("How many?")
    assert res.steps[0].is_error
    assert res.answer.startswith("Sorry")