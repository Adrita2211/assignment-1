"""Robustness (Taxonomy Axis 4, check 5): does the agent hold up under
rephrased or malformed input, not just the clean phrasing a fixture set was
originally tuned against?

Runs eval/fixtures.py's ROBUSTNESS_TICKETS -- typo'd, lowercase/terse,
all-caps, and over-verbose rephrasings of five real eval/fixtures.py TICKETS
entries (see each ticket's `derived_from`) -- through the exact same
pass/fail rule as eval/trajectory_eval.py: required_tools must be a subset
of the trajectory the agent actually produced. A robust agent should reach
the same trajectory as the clean original despite the noisy surface form;
this script's real value is comparing the two side by side, not just the
raw pass rate.

Usage:
    python -m eval.robustness_eval
"""
from __future__ import annotations

import asyncio
import os
import sys

from dotenv import load_dotenv

from eval._run_utils import run_ticket
from eval.fixtures import ROBUSTNESS_TICKETS, TICKETS
from eval.langfuse_scores import push_score

_BY_ID = {t["id"]: t for t in TICKETS}


async def run_suite() -> dict:
    results = []
    for ticket in ROBUSTNESS_TICKETS:
        outcome = await run_ticket(ticket["customer_id"], ticket["message"])
        required = ticket["required_tools"]
        actual = set(outcome["trajectory"])
        passed = required.issubset(actual)

        baseline = _BY_ID[ticket["derived_from"]]
        push_score(
            outcome["trace_id"], "robustness_pass", passed,
            comment=f"derived_from={ticket['derived_from']} required={sorted(required)} actual={outcome['trajectory']}",
        )
        results.append({
            "id": ticket["id"],
            "derived_from": ticket["derived_from"],
            "passed": passed,
            "required_tools": sorted(required),
            "actual_trajectory": outcome["trajectory"],
            "baseline_message": baseline["message"],
            "noisy_message": ticket["message"],
        })

    pass_count = sum(r["passed"] for r in results)
    return {"pass_rate": round(100 * pass_count / len(results), 1), "results": results}


def main():
    load_dotenv()
    if not os.environ.get("GROQ_API_KEY"):
        print("GROQ_API_KEY is not set -- cannot run the eval suite against a live model.")
        sys.exit(1)

    run = asyncio.run(run_suite())
    print(f"\nRobustness eval -- pass rate: {run['pass_rate']}%\n")
    for r in run["results"]:
        mark = "PASS" if r["passed"] else "FAIL"
        print(f"  [{mark}] {r['id']:<28} (vs {r['derived_from']})  required={r['required_tools']}  actual={r['actual_trajectory']}")
        if not r["passed"]:
            print(f"           noisy input: {r['noisy_message']!r}")

    if any(not r["passed"] for r in run["results"]):
        sys.exit(1)


if __name__ == "__main__":
    main()
