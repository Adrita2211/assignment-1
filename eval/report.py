"""Human-facing observability layer: one readable report that pulls every
other eval script's output into a single artifact "legible to a person who
wasn't there when it happened" -- a reviewer, a teammate, future-you three
weeks from now. Every other eval/*.py script prints to a terminal a person
has to have been watching live; this one writes eval/report_output.md, a
durable document.

Covers all six Taxonomy Axis 4 checks and the System + Longitudinal
observability layers in one run:

  Task Success / Groundedness  -> eval/llm_judge.py
  Policy Adherence             -> eval/policy_adherence_eval.py
  Safety                       -> eval/safety_eval.py
  Robustness                   -> eval/robustness_eval.py
  Calibration                  -> eval/calibration_eval.py
  System (latency/cost/errors) -> eval/trajectory_eval.py's per-run metrics
  Trajectory                   -> eval/trajectory_eval.py
  Longitudinal                 -> eval/run_history.jsonl, if present (this
                                   script does not itself append new runs --
                                   run eval/longitudinal_eval.py for that)

This is a real LLM-call-heavy run (every axis re-runs its own fixture set
against the live model) -- expect it to take a few minutes and use real
Groq quota. Use --skip to leave out sections while iterating.

Usage:
    python -m eval.report                              # full report
    python -m eval.report --skip safety calibration     # faster iteration
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

OUTPUT_PATH = Path(__file__).parent / "report_output.md"

SECTIONS = ["trajectory", "judge", "policy_adherence", "safety", "robustness", "calibration", "longitudinal"]


async def _run_trajectory():
    from eval.trajectory_eval import load_baseline, run_suite
    run = await run_suite(regressed=False)
    baseline = load_baseline()
    return run, baseline


async def _run_judge():
    from agent.provider import GroqProvider
    from eval.fixtures import TICKETS
    from eval.llm_judge import run_ticket as judge_one
    provider = GroqProvider()
    return [await judge_one(t, provider, n_runs=3) for t in TICKETS[:5]]


async def _run_policy_adherence():
    from eval.policy_adherence_eval import run_suite
    return await run_suite()


async def _run_safety():
    from eval.safety_eval import run_suite
    return await run_suite()


async def _run_robustness():
    from eval.robustness_eval import run_suite
    return await run_suite()


async def _run_calibration():
    from eval.calibration_eval import run_suite
    return await run_suite(n_runs=3)


def _load_longitudinal():
    from eval.longitudinal_eval import _load_history, analyze
    history = _load_history()
    return analyze(history) if history else None


def _render(sections: dict) -> str:
    lines = []
    lines.append(f"# Evaluation Report")
    lines.append(f"\nGenerated: {datetime.now(timezone.utc).isoformat()}")
    lines.append(
        "\nCovers Taxonomy Axis 4 (Task Success, Groundedness, Policy Adherence, "
        "Safety, Robustness, Calibration) and the System / Trajectory / "
        "Longitudinal observability layers. See each eval/*.py module for the "
        "scoring method behind each number below."
    )

    if "trajectory" in sections:
        run, baseline = sections["trajectory"]
        lines.append("\n## Trajectory (rule-based, gates CI)")
        lines.append(f"\nPass rate: **{run['pass_rate']}%** (baseline: {baseline['pass_rate']}%)")
        lines.append("\n| Ticket | Type | Result | Required | Actual |")
        lines.append("|---|---|---|---|---|")
        for r in run["results"]:
            mark = "PASS" if r["passed"] else "FAIL"
            lines.append(f"| {r['id']} | {r['ticket_type']} | {mark} | {r['required_tools']} | {r['actual_trajectory']} |")
        sm = run["system_metrics"]
        lines.append("\n### System layer (from this same run)")
        lines.append(
            f"\n- LLM calls: {sm['total_llm_calls']}\n"
            f"- Total latency: {sm['total_latency_s']}s (avg {sm['avg_latency_per_call_s']}s/call)\n"
            f"- Tokens: {sm['total_input_tokens']} in / {sm['total_output_tokens']} out\n"
            f"- Errors: {sm['error_count']} (rate {sm['error_rate']})"
        )

    if "judge" in sections:
        lines.append("\n## Groundedness + Task Success (LLM-as-judge, 3 runs/ticket)")
        lines.append("\n| Ticket | Groundedness (mean/stdev) | Task Success (mean/stdev) |")
        lines.append("|---|---|---|")
        for r in sections["judge"]:
            lines.append(
                f"| {r['id']} | {r['groundedness_mean']}/10 (±{r['groundedness_stdev']}) | "
                f"{r['task_success_mean']}/10 (±{r['task_success_stdev']}) |"
            )

    if "policy_adherence" in sections:
        run = sections["policy_adherence"]
        lines.append(f"\n## Policy Adherence\n\nPass rate: **{run['pass_rate']}%**")
        lines.append("\n| Ticket | Kind | Result | Boundary held | No refund confirmed |")
        lines.append("|---|---|---|---|---|")
        for r in run["results"]:
            mark = "PASS" if r["passed"] else "FAIL"
            lines.append(f"| {r['id']} | {r['kind']} | {mark} | {r['boundary_held']} | {r['no_refund_confirmed']} |")

    if "safety" in sections:
        run = sections["safety"]
        lines.append(f"\n## Safety (prompt injection / PII leakage)\n\nPass rate: **{run['pass_rate']}%**")
        lines.append("\n| Ticket | Result | Leaked | Boundary held | Refund override complied | Echoed system prompt |")
        lines.append("|---|---|---|---|---|---|")
        for r in run["results"]:
            mark = "PASS" if r["passed"] else "FAIL"
            lines.append(
                f"| {r['id']} | {mark} | {r['leaked_strings'] or '-'} | {r['boundary_held']} | "
                f"{r['complied_with_refund_override']} | {r['echoed_system_prompt']} |"
            )

    if "robustness" in sections:
        run = sections["robustness"]
        lines.append(f"\n## Robustness (typo'd / malformed / rephrased input)\n\nPass rate: **{run['pass_rate']}%**")
        lines.append("\n| Ticket | Derived from | Result | Required | Actual |")
        lines.append("|---|---|---|---|---|")
        for r in run["results"]:
            mark = "PASS" if r["passed"] else "FAIL"
            lines.append(f"| {r['id']} | {r['derived_from']} | {mark} | {r['required_tools']} | {r['actual_trajectory']} |")

    if "calibration" in sections:
        run = sections["calibration"]
        lines.append(f"\n## Calibration (honest gaps vs. fabrication)\n\nPass rate: **{run['pass_rate']}%**")
        lines.append("\n| Ticket | Result | Retrieval honest | Judge scores | Agreement |")
        lines.append("|---|---|---|---|---|")
        for r in run["results"]:
            mark = "PASS" if r["passed"] else "FAIL"
            lines.append(f"| {r['id']} | {mark} | {r['retrieval_honest']} | {r['judge_scores']} | {r['judge_agreement']} |")

    if "longitudinal" in sections:
        summary = sections["longitudinal"]
        lines.append("\n## Longitudinal (across recorded history)")
        if summary is None:
            lines.append(
                "\nNo recorded history yet -- run `python -m eval.longitudinal_eval --repeats N` "
                "at least once to start accumulating it."
            )
        else:
            lines.append(
                f"\n- Runs recorded: {summary['runs_recorded']}\n"
                f"- Mean pass rate: {summary['mean_pass_rate']}%\n"
                f"- Stdev pass rate: {summary['stdev_pass_rate']} points\n"
                f"- Drift (latest vs. prior): {summary['drift_vs_prior_runs']}\n"
                f"- Most-failed tickets: {summary['most_failed_tickets'] or '(none)'}\n"
                f"- Coverage gap: {summary['coverage_gap'] or '(none)'}"
            )

    return "\n".join(lines) + "\n"


async def main_async(skip: set[str]):
    sections = {}
    if "trajectory" not in skip:
        sections["trajectory"] = await _run_trajectory()
    if "judge" not in skip:
        sections["judge"] = await _run_judge()
    if "policy_adherence" not in skip:
        sections["policy_adherence"] = await _run_policy_adherence()
    if "safety" not in skip:
        sections["safety"] = await _run_safety()
    if "robustness" not in skip:
        sections["robustness"] = await _run_robustness()
    if "calibration" not in skip:
        sections["calibration"] = await _run_calibration()
    if "longitudinal" not in skip:
        sections["longitudinal"] = _load_longitudinal()

    report = _render(sections)
    OUTPUT_PATH.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nWritten to {OUTPUT_PATH}")


def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip", nargs="*", default=[], choices=SECTIONS, help="sections to leave out")
    args = parser.parse_args()

    if not os.environ.get("GROQ_API_KEY") and set(args.skip) != set(SECTIONS):
        print("GROQ_API_KEY is not set -- cannot run the eval suite against a live model.")
        sys.exit(1)

    asyncio.run(main_async(set(args.skip)))


if __name__ == "__main__":
    main()
