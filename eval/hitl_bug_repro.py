"""Regression test / reproduction for the real 'Double Refund' bug found
while building agent/hitl_store.py: v1's create_pending() had no
uniqueness check against resource_id, so two overlapping tickets against
the SAME order could each get their own PendingAction, both get approved
independently by two different reviewers who never see each other's
queue, and both execute -- two refunds issued for one order.

Run this against the CURRENT (fixed) agent/hitl_store.py and it should
print "BUG NOT REPRODUCIBLE (fixed)" -- the uniqueness guard in
create_pending() now raises DuplicatePendingActionError on the second
attempt. Kept as a permanent regression test, not a throwaway script, so
the fix can never silently regress.

Usage:
    python -m eval.hitl_bug_repro
"""
from __future__ import annotations

import json
from pathlib import Path

from agent.hitl import ApprovalStatus, DuplicatePendingActionError, new_pending_action
from agent.hitl_store import HITLStore

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_REPRO_DB_PATH = DATA_DIR / "hitl_store_repro.db"


def _fresh_store() -> HITLStore:
    _REPRO_DB_PATH.unlink(missing_ok=True)
    return HITLStore(db_path=_REPRO_DB_PATH)


def main():
    orders = json.loads((DATA_DIR / "orders.json").read_text(encoding="utf-8"))
    order = orders["ORD1007"]  # $399.00, over threshold, real over-threshold refund candidate
    store = _fresh_store()

    print("Simulating two overlapping tickets against the same order (ORD1007)...")
    action_a = new_pending_action(
        resource_id="ORD1007", ticket_id="ticket_A", customer_id="CUST005",
        amount_usd=order["order_total"], resource_snapshot=order,
    )
    action_b = new_pending_action(
        resource_id="ORD1007", ticket_id="ticket_B", customer_id="CUST005",
        amount_usd=order["order_total"], resource_snapshot=order,
    )

    try:
        store.create_pending(action_a)
        print(f"  ticket_A pending action created: {action_a.approval_id}")
        store.create_pending(action_b)
        print(f"  ticket_B pending action created: {action_b.approval_id}")
    except DuplicatePendingActionError as exc:
        print(f"\nBUG NOT REPRODUCIBLE (fixed) -- second create_pending() correctly raised: {exc}")
        return

    print("\nBUG REPRODUCED: two independent PendingActions exist for the same resource_id.")
    print("Approving both independently (as two different human reviewers would, unaware of each other)...")
    store.decide(action_a.approval_id, ApprovalStatus.APPROVED, decided_by="reviewer_1")
    store.decide(action_b.approval_id, ApprovalStatus.APPROVED, decided_by="reviewer_2")

    executed_a = store.mark_executed(action_a.approval_id)
    executed_b = store.mark_executed(action_b.approval_id)
    print(f"  ticket_A executed: {executed_a.status.value}")
    print(f"  ticket_B executed: {executed_b.status.value}")
    print(
        f"\nDOUBLE REFUND: ${action_a.amount_usd:.2f} was approved and executed TWICE "
        f"for order ORD1007 (total exposure: ${action_a.amount_usd * 2:.2f}), from a single real order."
    )


if __name__ == "__main__":
    main()
