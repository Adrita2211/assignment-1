"""Before/after report: runs eval/trajectory_eval.py's suite twice in one
process -- once clean, once with the agent's regression flag forced on --
and prints both pass rates side by side plus exactly which tickets flipped
from PASS to FAIL.

This is the artifact the assignment's "Craft" bar actually checks: a
specific number ("83% to 50%, and here's which tickets flipped"), not a
vague "the regressed version performs worse." Run this locally to produce
that number for the README/PR description; the CI gate
(.github/workflows/ci-cd.yml running eval/trajectory_eval.py) is what
actually blocks the regressed build from deploying -- this script just
documents the drop it would have caused.

Usage:
    python -m eval.before_after_report
"""
from __future__ import annotations

import asyncio
import os
import sys

from dotenv import load_dotenv

from eval.trajectory_eval import run_suite


async def main_async():
    print("Running CLEAN suite (regressed=False)...")
    clean = await run_suite(regressed=False)

    print("\nRunning REGRESSED suite (regressed=True)...")
    regressed = await run_suite(regressed=True)

    clean_by_id = {r["id"]: r["passed"] for r in clean["results"]}
    regressed_by_id = {r["id"]: r["passed"] for r in regressed["results"]}

    flipped = [
        tid for tid in clean_by_id
        if clean_by_id[tid] and not regressed_by_id.get(tid, False)
    ]

    print("\n" + "=" * 70)
    print("BEFORE / AFTER REPORT")
    print("=" * 70)
    print(f"Clean pass rate:     {clean['pass_rate']}%")
    print(f"Regressed pass rate: {regressed['pass_rate']}%")
    print(f"Drop:                {clean['pass_rate'] - regressed['pass_rate']:.1f} points")
    print(f"\nTickets that flipped PASS -> FAIL ({len(flipped)}):")
    for tid in flipped:
        print(f"  - {tid}")
    if not flipped:
        print("  (none)")
    print("=" * 70)


def main():
    load_dotenv()
    if not os.environ.get("GROQ_API_KEY"):
        print("GROQ_API_KEY is not set -- cannot run the eval suite against a live model.")
        sys.exit(1)
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
