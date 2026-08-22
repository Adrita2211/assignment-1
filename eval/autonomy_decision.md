# The autonomy decision: single-agent, not A2A (Assignment 3, §2.1)

**Path taken: B — write the justification.** Walking this agent through
Session 5's own four-question framework, against real tickets and real tool
calls from this codebase, the honest answer is no second agent is needed.

## The four questions, against this agent specifically

**1. Does any sub-task need meaningfully different tools, data, or authority
than the main agent already has?**

The standard candidate for this project is billing disputes ("this looks
like a chargeback/dispute, not a refund"). But this agent already has the
full authority chain a billing-dispute handler would need: `lookup_order`
and `check_account_status` (the same data a dispute review would start
from — order status, account standing, order history), `propose_refund_decision`
(a structured decision payload, §2.7), `agent/policy_boundary.py`'s
`evaluate_refund_policy()` (the actual authorization boundary — order
eligibility, amount-matches-record, account standing, the $150 threshold),
and the HITL gate (§2.5) for anything above that threshold. A genuine
"billing disputes" specialist would need *different* data (transaction/
dispute records this project's mock dataset doesn't model at all) and
*different* authority (reversing a charge, not just approving a refund) —
neither of which this ticket domain's actual tickets ever ask for. The
tickets that use words like "dispute" or "chargeback" in this project's
mock ticket history (`data/ticket_history.json`, CUST003's "flagged for a
suspected chargeback dispute" entry) are backward-looking account-standing
context, not a live task requiring dispute-specific tools.

**2. Can the candidate specialist's job fit in one honest sentence?**

Tried: *"Handles refund and compensation decisions above a policy
threshold."* That sentence already exists — it's exactly what §2.5's HITL
gate does, inside the existing agent, not a separate service. There's no
narrower, genuinely distinct one-sentence job left over once the policy
boundary and HITL gate are accounted for.

**3. Is there a real, current bottleneck — not just a tidier architecture
diagram?**

No. The full refund flow (propose → policy check → auto-approve or
HITL-pending → human decision → re-validated execution) already runs
end-to-end inside `agent/harness.py`'s single LangGraph, verified for real
in this assignment's Phase 3–5 testing (see the git history for the actual
captured runs: an auto-approved under-threshold refund, an over-threshold
refund correctly routed to HITL, a suspended-account rejection, a
mismatched-amount rejection). Splitting this into a separate process would
add a real network hop and a second deployment surface for zero capability
gain — nothing here is currently slow, currently wrong, or currently
blocked on the single-agent shape.

**4. Can you afford the failure mode?**

A second agent's failure mode is a dropped/malformed handoff — state that
should have crossed a process boundary and didn't, producing a confidently
wrong answer from incomplete context (this is exactly Assignment 3's own
named risk for teams that build the handoff, and the "found a real dropped
handoff" requirement Path A carries). Staying single-agent means that
specific failure mode simply doesn't exist here — the tradeoff is a
theoretical "more scalable/modular" argument against a concrete, avoidable
new class of bug this agent's actual ticket volume and complexity don't
justify taking on.

## The ticket that looks like it needs a handoff, resolved cleanly by the
## existing single agent

**Ticket:** CUST003 asks: *"My graphic tablet order ORD1008 never arrived,
it says lost in transit. I want a refund."*

This reads exactly like a "billing/compensation" ticket on first pass — it
mentions money, a specific dollar amount is at stake ($189.99), and the
order is over this project's $150 approval threshold, i.e. genuinely
non-trivial money, not an auto-approved small refund.

**What actually happened, real tool calls, single agent, no handoff:**

```
[CLASSIFY] ticket_type=delivery_issue
[RAG] retrieved: [('refund_eligibility', ...), ('lost_or_damaged_package', ...)]
[HARNESS] ALLOWED lookup_order({'order_id': 'ORD1008'})
[HARNESS] ALLOWED propose_refund_decision({
    'order_id': 'ORD1008', 'eligible': True,
    'reason': 'lost_or_damaged_package: orders marked "lost_in_transit" are
               automatically eligible for a full refund',
    'amount_usd': 189.99, 'requires_approval': True,
})
```

Response to the customer:

> Your refund request for order ORD1008 ($189.99) is above our review
> threshold and has been submitted for approval (reference:
> 0619405cf27547ecb8e852a0ea767087). We'll follow up once it's been
> reviewed.

The agent: classified the ticket, retrieved the two genuinely relevant
policy docs (grounding the decision in real policy text, not a guess),
looked up the real order, proposed a refund with the amount taken from the
tool-fetched `order_total` (not invented), had that proposal checked by the
policy boundary (which correctly flagged it as over-threshold), and
produced a real `PendingAction` via the HITL gate — all inside the single
`SupportHarness` session, no second process, no handoff, no dropped state.

It was later approved and genuinely executed:

```python
await resume_after_approval('0619405cf27547ecb8e852a0ea767087', 'approved', decided_by='reviewer_1')
# -> {'status': 'executed', 'refund_result': {'order_id': 'ORD1008', 'refund_issued': True, 'refunded_amount': 189.99}}
```

Nothing about this flow was blocked, slowed, or handled incorrectly by
staying single-agent. The "looks like it needs a specialist" instinct on
first read doesn't survive actually tracing what the ticket needed once you
look — exactly the discipline question 1 above is meant to force.
