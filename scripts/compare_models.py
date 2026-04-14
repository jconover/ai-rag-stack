#!/usr/bin/env python3
"""
Compare multiple LLMs on the same question through the Agentic RAG endpoint.

Runs one question against N Ollama models via POST /api/agent/chat and prints
the answers side-by-side plus a per-model breakdown of the agent_trace so you
can see WHERE each model succeeds or fails inside the agent loop
(plan / reflect / decide / verify).

Usage:
    # Default models, default question
    python scripts/compare_models.py

    # Your own question
    python scripts/compare_models.py -q "How do I roll back a Kubernetes deployment?"

    # Pick specific models
    python scripts/compare_models.py -m llama3.1:8b qwen2.5:32b deepseek-r1:32b

    # Point at a non-local backend
    python scripts/compare_models.py --host http://localhost:8000

Prerequisites:
    - Backend is running: `make start-dev`
    - Each model has been pulled: `docker exec ollama ollama pull <model>`
    - LLM_PROVIDER=ollama in your .env (the default)
"""

import argparse
import json
import sys
import time
from typing import Any, Dict, List

import requests

DEFAULT_MODELS = [
    "llama3.1:8b",
    "qwen2.5:14b",
    "qwen2.5:32b",
]

DEFAULT_QUESTION = (
    "How do I perform a rolling update of a Kubernetes deployment "
    "and roll it back if it fails?"
)


def run_one(host: str, question: str, model: str, timeout: int) -> Dict[str, Any]:
    """Send one question to /api/agent/chat with a specific model override."""
    url = f"{host.rstrip('/')}/api/agent/chat"
    payload = {"message": question, "model": model}
    t0 = time.time()
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        data["_elapsed_s"] = round(time.time() - t0, 2)
        data["_ok"] = True
        return data
    except requests.HTTPError as e:
        return {
            "_ok": False,
            "_elapsed_s": round(time.time() - t0, 2),
            "_error": f"HTTP {e.response.status_code}: {e.response.text[:200]}",
        }
    except Exception as e:
        return {
            "_ok": False,
            "_elapsed_s": round(time.time() - t0, 2),
            "_error": f"{type(e).__name__}: {e}",
        }


def summarize_trace(trace: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pull the interesting bits out of an agent_trace list."""
    summary = {
        "steps": len(trace),
        "plan_subqueries": None,
        "chunks_kept": 0,
        "chunks_dropped": 0,
        "retries": 0,
        "verified": None,
        "verify_reason": None,
    }
    for step in trace:
        name = step.get("step")
        if name == "plan":
            out = step.get("output") or {}
            sq = out.get("sub_queries") or out.get("subqueries") or out
            if isinstance(sq, list):
                summary["plan_subqueries"] = len(sq)
        elif name == "reflect":
            summary["chunks_kept"] += int(step.get("kept", 0) or 0)
            summary["chunks_dropped"] += int(step.get("dropped", 0) or 0)
        elif name == "decide" and step.get("action") == "retry":
            summary["retries"] += 1
        elif name == "verify":
            summary["verified"] = step.get("addresses_question")
            summary["verify_reason"] = step.get("reason")
    return summary


def print_header(text: str, ch: str = "=") -> None:
    print("\n" + ch * 78)
    print(text)
    print(ch * 78)


def print_result(model: str, result: Dict[str, Any]) -> None:
    print_header(f"MODEL: {model}")
    if not result.get("_ok"):
        print(f"  FAILED in {result.get('_elapsed_s')}s")
        print(f"  {result.get('_error')}")
        return

    print(f"  latency:       {result.get('_elapsed_s')}s")

    trace = result.get("agent_trace") or []
    s = summarize_trace(trace)
    print(f"  trace steps:   {s['steps']}")
    print(f"  sub-queries:   {s['plan_subqueries']}")
    print(f"  chunks kept:   {s['chunks_kept']}  (dropped {s['chunks_dropped']})")
    print(f"  retries:       {s['retries']}")
    print(f"  self-verified: {s['verified']}")
    if s["verify_reason"]:
        print(f"  verify reason: {s['verify_reason'][:140]}")

    meta = result.get("metadata") or {}
    if meta:
        print(f"  metadata:      {meta}")

    answer = (result.get("answer") or result.get("response") or "").strip()
    print("\n  --- ANSWER ---")
    for line in answer.splitlines()[:25]:
        print(f"  {line}")
    if len(answer.splitlines()) > 25:
        print(f"  ... ({len(answer.splitlines()) - 25} more lines)")


def print_matrix(question: str, results: Dict[str, Dict[str, Any]]) -> None:
    print_header("COMPARISON MATRIX", ch="#")
    print(f"Question: {question}\n")
    hdr = f"{'Model':<22} {'Time':>8} {'Steps':>6} {'Kept':>5} {'Drop':>5} {'Retry':>6} {'Verified':>9}"
    print(hdr)
    print("-" * len(hdr))
    for model, r in results.items():
        if not r.get("_ok"):
            print(f"{model:<22} {r.get('_elapsed_s', '-'):>8} {'ERR':>6}")
            continue
        s = summarize_trace(r.get("agent_trace") or [])
        print(
            f"{model:<22} "
            f"{r['_elapsed_s']:>7}s "
            f"{s['steps']:>6} "
            f"{s['chunks_kept']:>5} "
            f"{s['chunks_dropped']:>5} "
            f"{s['retries']:>6} "
            f"{str(s['verified']):>9}"
        )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("-q", "--question", default=DEFAULT_QUESTION)
    p.add_argument("-m", "--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--host", default="http://localhost:8000")
    p.add_argument("--timeout", type=int, default=300)
    p.add_argument("--save", metavar="FILE", help="Save raw results as JSON")
    args = p.parse_args()

    print(f"Question: {args.question}")
    print(f"Models:   {', '.join(args.models)}")
    print(f"Backend:  {args.host}")

    results: Dict[str, Dict[str, Any]] = {}
    for model in args.models:
        print(f"\n>>> Running {model} ...", flush=True)
        results[model] = run_one(args.host, args.question, model, args.timeout)
        print_result(model, results[model])

    print_matrix(args.question, results)

    if args.save:
        with open(args.save, "w") as f:
            json.dump({"question": args.question, "results": results}, f, indent=2)
        print(f"\nSaved raw results to {args.save}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
