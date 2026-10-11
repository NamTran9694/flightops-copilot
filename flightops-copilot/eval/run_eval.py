"""Evaluate the copilot.

Part 1 - Retrieval (no API key needed): hit@1, hit@3 and MRR on labeled queries.
Part 2 - Agent (needs ANTHROPIC_API_KEY): numeric accuracy against ground truth computed
         live from the database, correct tool selection, latency and token cost.

Usage:
    python eval/run_eval.py --retrieval-only
    python eval/run_eval.py
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flightops.db import run_query  # noqa: E402
from flightops.rag import DocIndex, default_embedder  # noqa: E402

GOLDEN = json.loads((ROOT / "eval" / "golden_set.json").read_text())
NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def eval_retrieval(index: DocIndex) -> dict:
    hits1 = hits3 = 0
    rr, misses = [], []
    for item in GOLDEN["retrieval"]:
        sources = [r["source"] for r in index.search(item["query"], k=5)]
        rank = sources.index(item["expected"]) + 1 if item["expected"] in sources else None
        hits1 += rank == 1
        hits3 += bool(rank and rank <= 3)
        rr.append(1 / rank if rank else 0.0)
        if rank != 1:
            misses.append({"query": item["query"], "expected": item["expected"], "got": sources[:3]})
    n = len(GOLDEN["retrieval"])
    return {"n": n, "hit@1": round(hits1 / n, 3), "hit@3": round(hits3 / n, 3),
            "mrr": round(sum(rr) / n, 3), "misses": misses}


def numbers_in(text: str) -> list[float]:
    out = []
    for m in NUM_RE.findall(text):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


def is_correct(answer: str, truth: float, rel_tol: float = 0.0, abs_tol: float = 0.0) -> bool:
    tol = max(abs(truth) * rel_tol, abs_tol, 1e-9)
    return any(abs(x - truth) <= tol for x in numbers_in(answer))


def eval_agent() -> dict:
    from flightops.agent import FlightOpsAgent
    agent = FlightOpsAgent()
    rows = []
    for item in GOLDEN["agent"]:
        truth = None
        if "ref_sql" in item:
            truth = float(run_query(item["ref_sql"])["rows"][0][0])
        elif "expected_value" in item:
            truth = float(item["expected_value"])
        res = agent.ask(item["question"])
        tools_used = [s.tool for s in res.steps]
        correct = None if truth is None else is_correct(res.answer, truth, item.get("rel_tol", 0.0),
                                                         item.get("abs_tol", 0.0))
        rows.append({
            "id": item["id"], "truth": None if truth is None else round(truth, 3),
            "correct": correct, "tool_ok": item["expected_tool"] in tools_used,
            "tools": tools_used, "errors": sum(s.is_error for s in res.steps),
            "seconds": res.seconds, "tokens": res.input_tokens + res.output_tokens,
            "answer": res.answer,
        })
        flag = "✅" if correct in (True, None) and rows[-1]["tool_ok"] else "❌"
        print(f"{flag} {item['id']:<18} truth={rows[-1]['truth']} tools={tools_used} {res.seconds}s")
    graded = [r for r in rows if r["correct"] is not None]
    return {
        "n": len(rows),
        "numeric_accuracy": round(sum(r["correct"] for r in graded) / len(graded), 3),
        "tool_selection_accuracy": round(sum(r["tool_ok"] for r in rows) / len(rows), 3),
        "median_seconds": statistics.median(r["seconds"] for r in rows),
        "avg_tokens": round(statistics.mean(r["tokens"] for r in rows)),
        "self_corrected_errors": sum(r["errors"] for r in rows),
        "details": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval-only", action="store_true")
    args = ap.parse_args()

    report = {"timestamp": datetime.now().isoformat(timespec="seconds")}
    report["retrieval"] = eval_retrieval(DocIndex.from_dir(embedder=default_embedder()))
    r = report["retrieval"]
    print(f"Retrieval: hit@1={r['hit@1']}  hit@3={r['hit@3']}  MRR={r['mrr']}  (n={r['n']})")

    if not args.retrieval_only:
        report["agent"] = eval_agent()
        a = report["agent"]
        print(f"Agent: numeric accuracy={a['numeric_accuracy']}  tool selection={a['tool_selection_accuracy']}  "
              f"median latency={a['median_seconds']}s  avg tokens={a['avg_tokens']}")

    out = ROOT / "eval" / "results.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
