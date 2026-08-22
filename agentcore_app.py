"""Bedrock AgentCore entrypoint (Assignment 3 §2.2) -- wraps the existing
SupportHarness in BedrockAgentCoreApp with a deliberately minimal
@app.entrypoint. This is the whole point: SupportHarness, its LangGraph
StateGraph, decide_node/validate_node/hitl_gate_node/act_node -- none of
it changes to fit this wrapper. If wrapping the agent had required
rewriting any of that, that would have been a sign of fighting the
abstraction instead of using it (the assignment's own words).

Confirmed against the real, currently-installed `bedrock-agentcore`
package (v1.22.0 at the time this was written -- this space moves fast,
per the assignment's own warning; re-check before relying on this):
`BedrockAgentCoreApp()`, the `@app.entrypoint` decorator (registers a
callable that receives "invocation payloads... passed unchanged" per the
library's own docstring), and `.run(port=8080)`.

Deliberately separate from server.py, not a replacement for it locally:
server.py is the Assignment 2 FastAPI entrypoint that ran on ECS; this is
the Assignment 3 entrypoint AgentCore's runtime invokes. Both wrap the
exact same SupportHarness -- the compute platform changes, the agent
logic doesn't.

Local dev / testing this file doesn't require AgentCore's cloud runtime
at all: `agentcore dev` (from bedrock-agentcore-starter-toolkit's CLI,
see README) runs this entrypoint locally with hot reload, invokable
against LLM_PROVIDER=groq exactly like server.py always has been. Only
`agentcore deploy` (the CLI's current name for what the assignment calls
`launch` -- renamed since the assignment was written, see README) needs
real AWS/Bedrock access.
"""
import uuid

from bedrock_agentcore import BedrockAgentCoreApp

from agent.harness import SupportHarness, resume_after_approval
from agent.tracing import langfuse

app = BedrockAgentCoreApp()


@app.entrypoint
async def invoke(payload: dict) -> dict:
    """Two payload shapes, dispatched on the presence of "action" --
    the ordinary chat turn (customer_id/message), and the HITL
    pause/resume step (Assignment 3 SS2.5), which is deliberately NOT part
    of a chat turn (see resume_after_approval's own docstring: a human
    decision arrives asynchronously, not as another message in the
    conversation). Both need to be reachable through the one deployed
    AgentCore endpoint for the "live and demonstrable against your deployed
    endpoint" requirement to actually cover HITL resume, not just the gate
    that creates the pending action."""
    if payload.get("action") == "resume_approval":
        result = await resume_after_approval(
            approval_id=payload["approval_id"],
            decision=payload["decision"],
            decided_by=payload.get("decided_by", "reviewer"),
        )
        return result

    customer_id = payload["customer_id"]
    message = payload["message"]
    ticket_id = payload.get("ticket_id") or uuid.uuid4().hex

    async with SupportHarness(customer_id, ticket_id=ticket_id) as harness:
        response = await harness.handle_turn(message)
        trajectory = harness.trajectory()
        trace_id = harness.last_trace_id

    langfuse.flush()  # same short-lived-request-cycle reasoning as server.py's /chat
    return {"response": response, "trajectory": trajectory, "trace_id": trace_id}


if __name__ == "__main__":
    app.run(port=8080)
