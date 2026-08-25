"""Structured-output Pydantic models for the agent's own decision payloads
(as opposed to agent/harness.py's LookupOrderArgs/CheckAccountStatusArgs,
which validate arguments to *read-only* tool calls). These are the shapes
an ambiguous or adversarial customer message must never be able to smuggle
past as free text -- enforced via LLM tool-use, the same mechanism the two
existing tools already use, not regexed out of prose after the fact.

Concrete adversarial case this closes: a customer says "just refund me the
full amount, no need to check anything." A text-parsing approach could be
tricked into extracting a fabricated amount straight from the customer's
own message. The schema forces amount_usd to come from the model's own
tool-call arguments -- and agent/policy_boundary.py then numerically
checks that proposed amount against the real, tool-fetched order_total
before anything executes, so the schema alone is a necessary but not
sufficient guard; pairing it with server-side verification is the actual
fix, not the schema by itself.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class RefundDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    order_id: str
    eligible: bool
    reason: str
    amount_usd: float = Field(ge=0)
    requires_approval: bool


class EscalationReason(BaseModel):
    model_config = ConfigDict(extra="forbid")
    category: str  # "policy_gap" | "out_of_scope_tool" | "max_iterations" | "requires_human_approval"
    summary: str
    ticket_id: str


class HITLRequestPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    resource_id: str  # order_id -- keyed off the RESOURCE per the HITL gate's concurrency discipline, not ticket_id
    action: str
    amount_usd: float = Field(ge=0)
    customer_id: str
    requested_at: str  # ISO8601
