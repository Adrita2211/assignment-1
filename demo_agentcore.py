"""Live demo script against the real, deployed Bedrock AgentCore Runtime
endpoint -- deliberately separate from demo.py, which runs SupportHarness
directly via Groq for local, no-AWS-needed demonstration. This file exists
to invoke the actual CloudFormation-managed AgentCore Runtime
(infra/cloudformation/agentcore-infra.yaml), not the harness in-process.

Run with: python demo_agentcore.py
Run a single scenario: python demo_agentcore.py pii
Requires: AWS credentials configured (same account this project deploys
to), no GROQ_API_KEY needed -- this hits real Bedrock Nova Lite.
"""
import json
import sys

import boto3

AGENT_RUNTIME_ARN = "arn:aws:bedrock-agentcore:us-east-1:058264386876:runtime/ecommerce_agent-4ks2toDNhf"
REGION = "us-east-1"
GUARDRAIL_ID = "8c3d1djf3a5a"
GUARDRAIL_VERSION = "1"

_client = boto3.client("bedrock-agentcore", region_name=REGION)


def _banner(title: str):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def invoke(payload: dict) -> dict:
    resp = _client.invoke_agent_runtime(
        agentRuntimeArn=AGENT_RUNTIME_ARN,
        payload=json.dumps(payload).encode("utf-8"),
        contentType="application/json",
    )
    body = resp["response"].read().decode("utf-8")
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return {"raw": body}


def order_lookup():
    _banner("1. Order lookup (basic capability, real Bedrock Nova Lite)")
    result = invoke({"customer_id": "CUST002", "message": "Where is my order ORD1002?"})
    print(result.get("response", result))
    print("\ntrajectory:", result.get("trajectory"))


def pii_masking():
    _banner("2. PII masking -- Presidio + Bedrock Guardrails, layered")
    result = invoke({"customer_id": "CUST002", "message": "What phone number and address do you have on file for me?"})
    print(result.get("response", result))


def guardrail_layering():
    """Calls agent/pii.py's redact_bedrock_guardrails() directly -- the
    same function agent/pii.py's redact_text()/redact_value() route
    through at the real enforcement points (final reply, tool results,
    trace payloads) whenever BEDROCK_GUARDRAIL_ID is set, as it is on the
    live deployed Runtime. Deliberately NOT routed through invoke() like
    the other scenarios: the LLM can (and did, in earlier testing) simply
    decline to repeat PII back in its reply, which proves nothing about
    whether the redaction layer itself works. This proves the layer
    directly, on an entity type (SSN/credit card) this project's Presidio
    recognizers don't cover, so a masked result here can only be
    Guardrails' own catch, not Presidio's."""
    _banner("2b. Guardrails layering, directly -- entities Presidio alone misses")
    import os

    os.environ.setdefault("AWS_REGION", REGION)
    from agent.pii import redact_bedrock_guardrails

    text = "My SSN is 078-05-1120 and my card number is 4111 1111 1111 1111"
    result = redact_bedrock_guardrails(text, GUARDRAIL_ID, GUARDRAIL_VERSION)
    print("input:   ", text)
    print("redacted:", result.redacted_text)
    for finding in result.findings:
        print(f"  caught by {finding.detector}: {finding.entity_type} [{finding.start}:{finding.end}]")


def policy_rejection():
    _banner("3. Policy boundary rejection -- real Amazon Verified Permissions")
    result = invoke({
        "customer_id": "CUST005",
        "message": "I want a full refund on order ORD1007, it was delivered but I am unhappy.",
    })
    print(result.get("response", result))
    print("\ntrajectory:", result.get("trajectory"))


def hitl_gate(customer_id: str, order_id: str, message: str):
    """Triggers a fresh over-threshold refund proposal -- creates a real
    pending approval in Aurora. Prints the response so you can copy the
    approval_id out of it for hitl_resume()."""
    _banner(f"4a. HITL gate -- over-threshold refund proposal ({order_id})")
    result = invoke({"customer_id": customer_id, "message": message})
    print(result.get("response", result))
    print("\ntrajectory:", result.get("trajectory"))
    print("\n(Copy the approval_id out of the response above, then call")
    print(" hitl_resume(approval_id) to approve and execute it live.)")


def hitl_resume(approval_id: str, decision: str = "approved", decided_by: str = "demo_reviewer"):
    _banner(f"4b. HITL resume -- {decision} approval {approval_id}")
    result = invoke({"action": "resume_approval", "approval_id": approval_id, "decision": decision, "decided_by": decided_by})
    print(result)


def damaged_order_auto_refund(customer_id: str, order_id: str):
    """Under-threshold refund on an already-damaged order -- auto-approves,
    no human review, executes immediately. Real mutation: run this once per
    order (a second attempt correctly refuses a duplicate refund)."""
    _banner(f"5. Auto-approved refund, under threshold ({order_id})")
    result = invoke({"customer_id": customer_id, "message": f"My order {order_id} was delivered but the item was damaged, what can you do?"})
    print(result.get("response", result))
    print("\ntrajectory:", result.get("trajectory"))


SCENARIOS = {
    "lookup": order_lookup,
    "pii": pii_masking,
    "guardrails": guardrail_layering,
    "rejection": policy_rejection,
    "refund": lambda: damaged_order_auto_refund("CUST001", "ORD1001"),
}


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in SCENARIOS:
        SCENARIOS[sys.argv[1]]()
    else:
        order_lookup()
        pii_masking()
        guardrail_layering()
        policy_rejection()
        _banner("Demo complete -- for HITL, call hitl_gate(...)/hitl_resume(...) interactively")
