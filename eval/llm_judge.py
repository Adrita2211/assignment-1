"""LLM-as-judge: a second scoring method for what a trajectory rule can't
verify. Scores two distinct Taxonomy-Axis-4 dimensions per ticket, from one
judge call each:

  - groundedness: is a specific claim in the response actually supported by
    the reference (the real policy doc text or order/account record), not
    just plausible-sounding.
  - task_success: independent of groundedness -- did the response actually
    address what the customer asked, in substance? A response can be
    perfectly grounded (every number it cites is real) and still fail the
    customer by being evasive, answering a different question than the one
    asked, or omitting the actual thing they needed to know. These two
    dimensions are scored separately on purpose: a rule/gate that only
    checks one would miss the other.

Deliberately a fact-checking rubric ("does every claim appear in the
reference, 0-10"), not a vague one ("rate this response's quality, 1-5").
Session 3's own finding (re-tested for real in agent-cicd-demo, see its
README) is that a vague rubric misses fabrication entirely; a specific one
catches it. Reuses whichever LLM the agent itself already uses (Groq) --
a judge call is just another prompt, not a different model.

Judge scores are non-deterministic -- run each ticket through the judge
N_RUNS times (default 3) and report the scores, not a single run treated as
fact; that's the whole reason this script takes multiple samples instead of
one.

Usage:
    python -m eval.llm_judge                 # judge the first 5 tickets
    python -m eval.llm_judge --tickets 8      # judge the first 8
    python -m eval.llm_judge --runs 5         # 5 judge samples per ticket
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys

from dotenv import load_dotenv

from pathlib import Path

from agent.harness import SupportHarness
from agent.provider import GroqProvider
from eval.fixtures import TICKETS
from eval.langfuse_scores import push_score

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
POLICY_DIR = Path(__file__).resolve().parent.parent / "policies"

JUDGE_PROMPT = """You are scoring a customer support agent's response on TWO \
separate dimensions. Be a strict fact-checker, not a helpfulness rater.

1. groundedness (0-10): is every specific claim in the response (numbers, \
dates, eligibility rules, order status, account standing) directly \
supported by the reference facts below? 10 = fully supported. 0 = the \
response states specific claims that are NOT in the reference facts at all \
(fabricated details), even if they sound plausible.

2. task_success (0-10): independent of groundedness -- did the response \
actually address what the customer asked, in substance? 10 = directly and \
completely answers the customer's actual question. 0 = evasive, answers a \
different question, or omits the thing the customer actually needed to \
know -- even if every word in it happens to be accurate.

Respond with ONLY a JSON object, nothing else, in exactly this shape:
{{"groundedness": <int 0-10>, "task_success": <int 0-10>}}

Customer question: {message}

Reference facts (ground truth): {reference}

Agent's response: {response}
"""


def _reference_for(ticket: dict) -> str:
    """Ground truth text for the judge to fact-check against -- pulled
    directly from this repo's own mock data / policy docs, never from
    anything the agent itself said, so the judge can't be fooled by an
    agent that fabricates confidently."""
    required = ticket["required_tools"]
    if "retrieve_policy" in required:
        # Best-effort: concatenate every policy doc so the judge always has
        # the real text available, rather than re-running retrieval (which
        # would just reproduce whatever bug we're trying to catch).
        return "\n\n".join(p.read_text(encoding="utf-8") for p in sorted(POLICY_DIR.glob("*.md")))
    if "lookup_order" in required:
        orders = json.loads((DATA_DIR / "orders.json").read_text(encoding="utf-8"))
        for order_id, record in orders.items():
            if order_id in ticket["message"]:
                return json.dumps(record)
        return json.dumps(orders)
    if "check_account_status" in required:
        accounts = json.loads((DATA_DIR / "accounts.json").read_text(encoding="utf-8"))
        return json.dumps(accounts.get(ticket["customer_id"], accounts))
    return ""


def judge_response(provider: GroqProvider, ticket: dict, response_text: str) -> dict:
    """One judge call, both dimensions. Falls back to 0/0 on a malformed
    (non-JSON) judge reply rather than raising -- a judge parsing failure
    should show up as a low score to investigate, not crash the whole suite."""
    reference = _reference_for(ticket)
    prompt = JUDGE_PROMPT.format(message=ticket["message"], reference=reference, response=response_text)
    result = provider.call([
        {"role": "system", "content": "You are a precise, terse fact-checking judge. Reply with JSON only."},
        {"role": "user", "content": prompt},
    ])
    raw = (result["text"] or "").strip()
    try:
        # Judges sometimes wrap JSON in prose or a code fence despite
        # instructions -- extract the {...} span rather than requiring an
        # exact-match parse.
        start, end = raw.index("{"), raw.rindex("}") + 1
        parsed = json.loads(raw[start:end])
        groundedness = max(0, min(10, int(parsed.get("groundedness", 0))))
        task_success = max(0, min(10, int(parsed.get("task_success", 0))))
    except (ValueError, TypeError, KeyError):
        groundedness, task_success = 0, 0
    return {"groundedness": groundedness, "task_success": task_success}


async def run_ticket(ticket: dict, judge_provider: GroqProvider, n_runs: int) -> dict:
    async with SupportHarness(ticket["customer_id"], ticket_id=ticket["id"]) as harness:
        response = await harness.handle_turn(ticket["message"])
        trace_id = harness.last_trace_id

    scores = [judge_response(judge_provider, ticket, response) for _ in range(n_runs)]
    groundedness_scores = [s["groundedness"] for s in scores]
    task_success_scores = [s["task_success"] for s in scores]
    groundedness_mean = round(statistics.fmean(groundedness_scores), 2)
    groundedness_stdev = round(statistics.pstdev(groundedness_scores), 2) if len(groundedness_scores) > 1 else 0.0
    task_success_mean = round(statistics.fmean(task_success_scores), 2)
    task_success_stdev = round(statistics.pstdev(task_success_scores), 2) if len(task_success_scores) > 1 else 0.0

    push_score(
        trace_id, "groundedness", groundedness_mean, data_type="NUMERIC",
        comment=f"runs={groundedness_scores} stdev={groundedness_stdev} (0-10, fact-checked against reference)",
    )
    push_score(
        trace_id, "task_success", task_success_mean, data_type="NUMERIC",
        comment=f"runs={task_success_scores} stdev={task_success_stdev} (0-10, independent of groundedness)",
    )

    return {
        "id": ticket["id"],
        "response": response,
        "groundedness_scores": groundedness_scores,
        "groundedness_mean": groundedness_mean,
        "groundedness_stdev": groundedness_stdev,
        "task_success_scores": task_success_scores,
        "task_success_mean": task_success_mean,
        "task_success_stdev": task_success_stdev,
    }


async def run_suite(n_tickets: int, n_runs: int) -> list[dict]:
    judge_provider = GroqProvider()
    tickets = TICKETS[:n_tickets]
    return [await run_ticket(t, judge_provider, n_runs) for t in tickets]


def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickets", type=int, default=5, help="how many fixed tickets to judge (>=5 required by the assignment)")
    parser.add_argument("--runs", type=int, default=3, help="judge samples per ticket (non-determinism check)")
    args = parser.parse_args()

    if not os.environ.get("GROQ_API_KEY"):
        print("GROQ_API_KEY is not set -- cannot run the judge against a live model.")
        sys.exit(1)

    results = asyncio.run(run_suite(args.tickets, args.runs))

    print(f"\nLLM-as-judge (groundedness + task_success, 0-10 each) over {len(results)} tickets, {args.runs} runs each:\n")
    for r in results:
        print(
            f"  {r['id']:<30} groundedness={r['groundedness_scores']} (mean={r['groundedness_mean']}, "
            f"stdev={r['groundedness_stdev']})  task_success={r['task_success_scores']} "
            f"(mean={r['task_success_mean']}, stdev={r['task_success_stdev']})"
        )

    overall_groundedness = round(statistics.fmean(r["groundedness_mean"] for r in results), 2)
    overall_task_success = round(statistics.fmean(r["task_success_mean"] for r in results), 2)
    print(f"\nOverall average groundedness:  {overall_groundedness}/10")
    print(f"Overall average task_success: {overall_task_success}/10")


if __name__ == "__main__":
    main()
