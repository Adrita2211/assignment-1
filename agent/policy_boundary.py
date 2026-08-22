"""A real authorization boundary on refund amounts, enforced outside the
prompt -- the same non-negotiable pattern as agent/harness.py's harness
boundary (validate_node, between decide and act), one layer further out:
not "is this call permitted at all" but "is this specific amount, for
this specific customer, within policy."

Hand-rolled, Cedar-shaped, chosen as the documented fallback per this
assignment's own explicitly-permitted alternative to Amazon Verified
Permissions: a small, explicit, auditable rule set (not scattered if
statements), with each check commented against what the equivalent Cedar
`permit(...) when {...}` statement would say, so migrating to real AVP
later is substituting the evaluator behind evaluate_refund_policy(), not
redesigning the check itself.

Plugs into agent/harness.py's validate_node as a fifth check layer,
specific to the propose_refund_decision tool, running AFTER the existing
four-layer ownership check (tool allowlist, schema, ID format, ownership)
already passes.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from agent.schemas import RefundDecision

REFUND_APPROVAL_THRESHOLD_USD = 150.00  # matches the number named in policies/refund_eligibility.md

_REFUND_ELIGIBLE_STATUSES = {"delivered", "delivered_damaged", "lost_in_transit"}


class PolicyDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    allowed: bool
    reason: str
    requires_approval: bool


def evaluate_refund_policy(*, order: dict, proposed: RefundDecision, customer_account: dict) -> PolicyDecision:
    """Cedar-shaped rule evaluation. Each check below is the Python
    equivalent of one Cedar `permit`/`forbid` statement -- kept as
    explicit, ordered conditionals rather than one boolean, so a rejection
    always names exactly which rule fired (needed for the audit log and
    for the "must demonstrate live rejection" requirement)."""

    # Cedar equivalent: forbid(...) when { order.status not in eligible_statuses };
    if order["status"] not in _REFUND_ELIGIBLE_STATUSES:
        return PolicyDecision(
            allowed=False,
            reason=f"order status {order['status']!r} is not refund-eligible (must be one of {sorted(_REFUND_ELIGIBLE_STATUSES)})",
            requires_approval=False,
        )

    # Cedar equivalent: forbid(...) when { proposed.amount_usd != order.order_total };
    # The model's proposed amount is never trusted as ground truth -- this
    # is the server-side check the structured-output schema alone can't
    # provide (see agent/schemas.py's RefundDecision docstring).
    if abs(proposed.amount_usd - order["order_total"]) > 0.01:
        return PolicyDecision(
            allowed=False,
            reason=f"proposed amount ${proposed.amount_usd:.2f} does not match order total on record (${order['order_total']:.2f})",
            requires_approval=False,
        )

    # Cedar equivalent: forbid(...) when { customer.standing == "suspended" };
    if customer_account["standing"] == "suspended":
        return PolicyDecision(
            allowed=False,
            reason="suspended accounts cannot receive refunds",
            requires_approval=False,
        )

    # Cedar equivalent: permit(...) when { order.order_total >= threshold } advice requires_approval;
    if order["order_total"] >= REFUND_APPROVAL_THRESHOLD_USD:
        return PolicyDecision(
            allowed=True,
            reason=f"order total ${order['order_total']:.2f} is at or above the ${REFUND_APPROVAL_THRESHOLD_USD:.2f} approval threshold",
            requires_approval=True,
        )

    # Cedar equivalent: permit(...) when { order.order_total < threshold };
    return PolicyDecision(
        allowed=True,
        reason=f"order total ${order['order_total']:.2f} is under the ${REFUND_APPROVAL_THRESHOLD_USD:.2f} threshold, auto-approvable",
        requires_approval=False,
    )
