"""The regression gate. Runs the fixed eval set (eval/fixtures.py) against a
fresh SupportHarness per ticket, scores each one by whether the trajectory it
actually produced (SupportHarness.trajectory() -- retrieve_policy /
lookup_order / check_account_status) is a superset of that ticket's
required_tools, and compares the aggregate pass rate against the stored
baseline (eval/baseline.json).

Deliberately scores the *path*, not the final-answer text: an agent that
answers a policy question confidently from memory instead of grounding it can
produce a final answer that reads fine while the trajectory reveals the
grounding step never ran -- exactly the defect AGENT_REGRESSED=true injects
(see agent/harness.py, retrieve_node).

Relative regression, not an absolute bar: fails only if the score drops more
than --threshold points below baseline, not simply if it's under some fixed
number. An absolute threshold lets a score drift downward forever as long as
it never crosses the line; a relative check catches the drift itself.

Usage:
    python -m eval.trajectory_eval                    # run the gate against baseline.json
    python -m eval.trajectory_eval --update-baseline   # (re)write baseline.json
    AGENT_REGRESSED=true python -m eval.trajectory_eval  # demo the gate failing
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from agent.harness import SupportHarness
from eval.fixtures import TICKETS
from eval.langfuse_scores import push_score

BASELINE_PATH = Path(__file__).parent / "baseline.json"
DEFAULT_THRESHOLD_POINTS = 15


async def run_ticket(ticket: dict, regressed: bool | None = None) -> dict:
    async with SupportHarness(ticket["customer_id"], regressed=regressed, ticket_id=ticket["id"]) as harness:
        response = await harness.handle_turn(ticket["message"])
        trajectory = harness.trajectory()
        metrics = harness.metrics()  # System-layer signal, piggybacked on this same run
        trace_id = harness.last_trace_id

    required = ticket["required_tools"]
    passed = required.issubset(set(trajectory))

    push_score(
        trace_id, "trajectory_pass", passed,
        comment=f"required={sorted(required)} actual={trajectory}",
    )
    if metrics["llm_calls"]:
        push_score(trace_id, "efficiency_latency_s", metrics["total_latency_s"], data_type="NUMERIC")
        push_score(
            trace_id, "efficiency_total_tokens",
            metrics["total_input_tokens"] + metrics["total_output_tokens"], data_type="NUMERIC",
        )

    return {
        "id": ticket["id"],
        "ticket_type": ticket["ticket_type"],
        "passed": passed,
        "required_tools": sorted(required),
        "actual_trajectory": trajectory,
        "response": response,
        "metrics": metrics,
    }


def _aggregate_system_metrics(results: list[dict]) -> dict:
    """Sums each ticket's SupportHarness.metrics() into one suite-level
    System-observability snapshot (latency, tokens, error rate) -- so a
    caller (eval/report.py) gets a real cost/latency/error number for this
    exact suite run without paying for a second pass over the model."""
    all_metrics = [r["metrics"] for r in results]
    total_calls = sum(m["llm_calls"] for m in all_metrics)
    total_errors = sum(m["error_count"] for m in all_metrics)
    total_latency = sum(m["total_latency_s"] for m in all_metrics)
    return {
        "total_llm_calls": total_calls,
        "total_latency_s": round(total_latency, 3),
        "avg_latency_per_call_s": round(total_latency / total_calls, 3) if total_calls else 0.0,
        "total_input_tokens": sum(m["total_input_tokens"] for m in all_metrics),
        "total_output_tokens": sum(m["total_output_tokens"] for m in all_metrics),
        "error_count": total_errors,
        "error_rate": round(total_errors / total_calls, 3) if total_calls else 0.0,
    }


async def run_suite(regressed: bool | None = None) -> dict:
    """regressed=None defers to each SupportHarness's own default (the
    AGENT_REGRESSED env var) -- what the CI gate and a real deploy use.
    Pass True/False explicitly to force a variant regardless of the
    environment, which is how eval/before_after_report.py compares both in
    one process."""
    results = [await run_ticket(t, regressed=regressed) for t in TICKETS]
    pass_count = sum(r["passed"] for r in results)
    pass_rate = round(100 * pass_count / len(results), 1)
    return {"pass_rate": pass_rate, "results": results, "system_metrics": _aggregate_system_metrics(results)}


def load_baseline() -> dict:
    if not BASELINE_PATH.exists():
        return {"pass_rate": 0.0}
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--update-baseline", action="store_true")
    parser.add_argument(
        "--threshold", type=float,
        default=float(os.environ.get("REGRESSION_THRESHOLD_POINTS", DEFAULT_THRESHOLD_POINTS)),
    )
    args = parser.parse_args()

    if not os.environ.get("GROQ_API_KEY"):
        print("GROQ_API_KEY is not set -- cannot run the eval suite against a live model.")
        sys.exit(1)

    run = asyncio.run(run_suite())

    print(f"\nTrajectory eval -- pass rate: {run['pass_rate']}%\n")
    for r in run["results"]:
        mark = "PASS" if r["passed"] else "FAIL"
        print(f"  [{mark}] {r['id']:<30} type={r['ticket_type']:<20} required={r['required_tools']} actual={r['actual_trajectory']}")

    if args.update_baseline:
        BASELINE_PATH.write_text(json.dumps({"pass_rate": run["pass_rate"]}, indent=2) + "\n", encoding="utf-8")
        print(f"\nBaseline updated: {run['pass_rate']}%")
        return

    baseline = load_baseline()
    drop = baseline["pass_rate"] - run["pass_rate"]
    print(
        f"\nBaseline: {baseline['pass_rate']}%  |  Current: {run['pass_rate']}%  |  "
        f"Drop: {drop:.1f} points  |  Threshold: {args.threshold} points"
    )

    if drop > args.threshold:
        print(f"\nREGRESSION GATE: FAILED -- dropped {drop:.1f} points, exceeds {args.threshold}-point threshold.")
        sys.exit(1)

    print("\nREGRESSION GATE: PASSED")


if __name__ == "__main__":
    main()
