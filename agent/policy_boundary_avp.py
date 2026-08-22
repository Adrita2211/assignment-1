"""Amazon Verified Permissions-backed refund policy boundary -- the real
managed-Cedar-service implementation, replacing agent/policy_boundary.py's
hand-rolled fallback now that AVP is actually provisioned (Assignment 3
§2.6). Same public shape (evaluate_refund_policy_avp returns a
PolicyDecision with the identical allowed/reason/requires_approval fields)
so agent/harness.py's validate_node needs only a call-site swap, not a
redesign.

Policy store: KozP4Mk6man7ivCYxeqBMP (us-east-1). Schema (RefundPolicy
namespace) and four static Cedar policies, verified live against all four
branches before this wrapper was written (see README's policy-boundary
section for the real is-authorized output on each):
  - GDw11cxJqYGZEKBjGN4TZT  forbid when order_status not refund-eligible
  - LugUCiwxL7ovzd5N4uVsPw  forbid when proposed amount != order_total
  - MKd6FZzur6QhKaAwuzYnGM  forbid when account_suspended
  - FbncATAgvRbEb8YDAxJesz  baseline permit (Cedar: explicit forbid always
    overrides permit, so this only takes effect once none of the three
    forbids fire)

What AVP gives for free that the hand-rolled version had to build by hand:
the actual allow/deny decision and *which* policy fired
(determiningPolicies), as a real authorization-service call, auditable and
versionable independently of this codebase. What AVP does NOT give for
free, and why this module still exists rather than a bare is-authorized
call: Cedar's decision is ALLOW/DENY, not a human-readable reason string --
the request-side context values are already known here (this module built
them), so producing the reason text is trivial, but it is application
logic layered on top of Cedar, not something IsAuthorized returns.

The requires_approval flag is deliberately NOT a Cedar permit/forbid
distinction -- being over the refund threshold isn't a policy violation,
it's a routing decision on top of an already-permitted request (the same
design the hand-rolled version uses; see its module docstring). Computed
here from the same real order_total the amount-match check already
verified, not asked of Cedar.
"""
from __future__ import annotations

import os

from agent.policy_boundary import PolicyDecision, REFUND_APPROVAL_THRESHOLD_USD
from agent.schemas import RefundDecision

_REFUND_ELIGIBLE_STATUSES = {"delivered", "delivered_damaged", "lost_in_transit"}

DEFAULT_POLICY_STORE_ID = "KozP4Mk6man7ivCYxeqBMP"

_POLICY_ID_TO_REASON = {
    "GDw11cxJqYGZEKBjGN4TZT": "status",
    "LugUCiwxL7ovzd5N4uVsPw": "amount",
    "MKd6FZzur6QhKaAwuzYnGM": "suspended",
}

_client = None


def _get_client(region: str | None = None):
    global _client
    if _client is None:
        import boto3

        _client = boto3.client("verifiedpermissions", region_name=region or os.environ.get("AWS_REGION", "us-east-1"))
    return _client


def evaluate_refund_policy_avp(
    *, order: dict, proposed: RefundDecision, customer_account: dict,
    policy_store_id: str | None = None, region: str | None = None,
) -> PolicyDecision:
    store_id = policy_store_id or os.environ.get("AVP_POLICY_STORE_ID", DEFAULT_POLICY_STORE_ID)
    amount_matches = abs(proposed.amount_usd - order["order_total"]) <= 0.01

    response = _get_client(region).is_authorized(
        policyStoreId=store_id,
        principal={"entityType": "RefundPolicy::Customer", "entityId": customer_account["customer_id"]},
        action={"actionType": "RefundPolicy::Action", "actionId": "ProposeRefund"},
        resource={"entityType": "RefundPolicy::Order", "entityId": order["order_id"]},
        context={
            "contextMap": {
                "order_status": {"string": order["status"]},
                "amount_matches": {"boolean": amount_matches},
                "account_suspended": {"boolean": customer_account["standing"] == "suspended"},
            }
        },
    )

    if response["decision"] == "DENY":
        fired = [p["policyId"] for p in response.get("determiningPolicies", [])]
        category = next((_POLICY_ID_TO_REASON[p] for p in fired if p in _POLICY_ID_TO_REASON), "unknown")
        reason = {
            "status": f"order status {order['status']!r} is not refund-eligible (must be one of {sorted(_REFUND_ELIGIBLE_STATUSES)})",
            "amount": f"proposed amount ${proposed.amount_usd:.2f} does not match order total on record (${order['order_total']:.2f})",
            "suspended": "suspended accounts cannot receive refunds",
            "unknown": f"denied by Verified Permissions (policies: {fired})",
        }[category]
        return PolicyDecision(allowed=False, reason=reason, requires_approval=False)

    requires_approval = order["order_total"] >= REFUND_APPROVAL_THRESHOLD_USD
    reason = (
        f"order total ${order['order_total']:.2f} is at or above the ${REFUND_APPROVAL_THRESHOLD_USD:.2f} approval threshold"
        if requires_approval
        else f"order total ${order['order_total']:.2f} is under the ${REFUND_APPROVAL_THRESHOLD_USD:.2f} threshold, auto-approvable"
    )
    return PolicyDecision(allowed=True, reason=reason, requires_approval=requires_approval)
