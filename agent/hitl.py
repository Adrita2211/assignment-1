"""HITL approval gate: two genuinely distinct mechanisms, not one feature
doing double duty.

1. The approval gate -- an action above REFUND_APPROVAL_THRESHOLD_USD
   (agent/policy_boundary.py) produces a PendingAction instead of
   executing.
2. Pause/resume -- the agent halts, a real snapshot (not just a boolean)
   persists into agent/hitl_store.py's HITLStore, and execution resumes
   from that point once a human decides, re-validating live state first
   (see HITLStore.revalidate_before_execution's docstring for the found
   real bug this exists to prevent).

An explicit state machine, not a boolean, with a real expiry window.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum

from pydantic import BaseModel, ConfigDict

APPROVAL_WINDOW = timedelta(hours=24)


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    EXECUTED = "executed"


class PendingAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    approval_id: str
    resource_id: str        # order_id -- concurrency/uniqueness keys off THIS, not ticket_id or approval_id
    ticket_id: str
    customer_id: str
    action: str              # "issue_refund"
    amount_usd: float
    status: ApprovalStatus
    created_at: str          # ISO8601
    expires_at: str          # ISO8601, created_at + APPROVAL_WINDOW
    resource_snapshot: dict  # order dict AS FETCHED at request time -- re-validated on resume, never trusted stale
    decided_at: str | None = None
    decided_by: str | None = None


def new_pending_action(*, resource_id: str, ticket_id: str, customer_id: str, amount_usd: float, resource_snapshot: dict) -> PendingAction:
    now = datetime.now(timezone.utc)
    return PendingAction(
        approval_id=uuid.uuid4().hex,
        resource_id=resource_id,
        ticket_id=ticket_id,
        customer_id=customer_id,
        action="issue_refund",
        amount_usd=amount_usd,
        status=ApprovalStatus.PENDING,
        created_at=now.isoformat(),
        expires_at=(now + APPROVAL_WINDOW).isoformat(),
        resource_snapshot=resource_snapshot,
    )


class DuplicatePendingActionError(Exception):
    """Raised when create_pending() is asked to open a second active
    PendingAction against a resource_id that already has one -- the fix
    for the 'Double Refund' bug (see agent/hitl_store.py)."""


class StaleStateError(Exception):
    """Raised when a resume/execute call's re-fetched live order state
    doesn't match the resource_snapshot the approval was originally
    granted against -- refuse to proceed silently on state that's changed
    since the human approved it."""


class ApprovalExpiredError(Exception):
    """Raised when deciding or executing a PendingAction whose expires_at
    has passed."""


class ApprovalAlreadyDecidedError(Exception):
    """Raised when deciding or executing a PendingAction that's no longer
    in PENDING status -- an idempotency guard against a replayed/duplicate
    decide call."""
