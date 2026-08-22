"""Shared "run one ticket through a fresh SupportHarness and capture
everything an eval script might need" helper, used by trajectory_eval.py,
policy_adherence_eval.py, safety_eval.py, robustness_eval.py, and
calibration_eval.py so each of those stays focused on its own scoring rule
instead of re-deriving the same harness plumbing five times.
"""
from __future__ import annotations

from agent.harness import SupportHarness


async def run_ticket(customer_id: str, message: str, regressed: bool | None = None, ticket_id: str = "unscoped") -> dict:
    async with SupportHarness(customer_id, regressed=regressed, ticket_id=ticket_id) as harness:
        response = await harness.handle_turn(message)
        return {
            "response": response,
            "trajectory": harness.trajectory(),
            "audit_log": list(harness.audit_log),
            "retrieved_doc_ids": list(harness.last_retrieved_doc_ids),
            "metrics": harness.metrics(),
            "trace_id": harness.last_trace_id,
        }
