"""Longitudinal observability layer: drift, variance, failure clustering,
and coverage -- across many runs of the eval suite, not a single snapshot.
A single trajectory_eval run answers "did it pass right now"; this answers
"is it getting worse, and is that concentrated in one place."

Runs eval/trajectory_eval.py's suite --repeats times, appends each run's
per-ticket result to eval/run_history.jsonl (one JSON line per run, never
overwritten -- history accumulates across every invocation, not just this
one), then reports on the accumulated history:

  - drift: is the most recent run's pass rate trending down relative to the
    history's mean (a single run below baseline is expected noise; a
    consistent downward trend across the recorded history is drift).
  - variance: stdev of pass rate across all recorded runs -- a proxy for how
    much a single run can be trusted at face value.
  - failure clustering: which specific ticket IDs fail most often across
    history, not just the aggregate rate -- a rate that looks stable in
    aggregate can still be one ticket failing every single time.
  - coverage: how many of the fixed ticket set's IDs have ever actually
    been exercised in the recorded history (should be all of them, every
    run -- this catches a suite that silently stopped running some tickets).

Usage:
    python -m eval.longitudinal_eval --repeats 5    # append 5 fresh runs, then report
    python -m eval.longitudinal_eval --repeats 0    # report only, no new runs
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from eval.fixtures import TICKETS
from eval.trajectory_eval import run_suite

HISTORY_PATH = Path(__file__).parent / "run_history.jsonl"
ALL_TICKET_IDS = {t["id"] for t in TICKETS}


async def _record_runs(n: int) -> None:
    with HISTORY_PATH.open("a", encoding="utf-8") as f:
        for i in range(n):
            run = await run_suite(regressed=False)
            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "pass_rate": run["pass_rate"],
                "failed_ids": [r["id"] for r in run["results"] if not r["passed"]],
                "exercised_ids": [r["id"] for r in run["results"]],
            }
            f.write(json.dumps(record) + "\n")
            print(f"  run {i + 1}/{n}: pass_rate={run['pass_rate']}%  failed={record['failed_ids'] or '(none)'}")


def _load_history() -> list[dict]:
    if not HISTORY_PATH.exists():
        return []
    lines = HISTORY_PATH.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def analyze(history: list[dict]) -> dict:
    pass_rates = [r["pass_rate"] for r in history]
    mean_rate = round(statistics.fmean(pass_rates), 2)
    stdev_rate = round(statistics.pstdev(pass_rates), 2) if len(pass_rates) > 1 else 0.0

    # Drift: compare the most recent run against the mean of everything
    # BEFORE it, not against itself -- otherwise every history trivially
    # "drifts" by definition.
    drift = None
    if len(pass_rates) >= 2:
        prior_mean = round(statistics.fmean(pass_rates[:-1]), 2)
        drift = round(pass_rates[-1] - prior_mean, 2)

    failure_counts = Counter()
    for r in history:
        failure_counts.update(r["failed_ids"])

    exercised_ever = set()
    for r in history:
        exercised_ever.update(r["exercised_ids"])
    coverage_gap = sorted(ALL_TICKET_IDS - exercised_ever)

    return {
        "runs_recorded": len(history),
        "mean_pass_rate": mean_rate,
        "stdev_pass_rate": stdev_rate,
        "drift_vs_prior_runs": drift,
        "most_failed_tickets": failure_counts.most_common(5),
        "coverage_gap": coverage_gap,
    }


def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3, help="how many fresh runs to append before reporting")
    args = parser.parse_args()

    if args.repeats > 0:
        if not os.environ.get("GROQ_API_KEY"):
            print("GROQ_API_KEY is not set -- cannot run the eval suite against a live model.")
            sys.exit(1)
        print(f"Recording {args.repeats} fresh run(s) to {HISTORY_PATH.name}...")
        asyncio.run(_record_runs(args.repeats))

    history = _load_history()
    if not history:
        print("\nNo history recorded yet -- run with --repeats > 0 at least once.")
        return

    summary = analyze(history)
    print("\n" + "=" * 70)
    print("LONGITUDINAL REPORT")
    print("=" * 70)
    print(f"Runs recorded:        {summary['runs_recorded']}")
    print(f"Mean pass rate:       {summary['mean_pass_rate']}%")
    print(f"Stdev pass rate:      {summary['stdev_pass_rate']} points")
    drift = summary["drift_vs_prior_runs"]
    if drift is not None:
        direction = "down" if drift < 0 else "up" if drift > 0 else "flat"
        print(f"Drift (latest vs prior runs): {drift:+.2f} points ({direction})")
    else:
        print("Drift: not enough history yet (need >= 2 runs)")
    print(f"Most-failed tickets:  {summary['most_failed_tickets'] or '(none recorded)'}")
    print(f"Coverage gap:         {summary['coverage_gap'] or '(none -- every fixed ticket has been exercised)'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
