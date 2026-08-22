"""HTTP entrypoint for the deployed agent (ECS Fargate behind an ALB).

Deliberately separate from main.py: main.py is a stateful interactive CLI
session for one customer_id across many turns; this server is stateless per
request (a fresh SupportHarness -- and a fresh MCP server subprocess bound to
that request's customer_id -- for every call), which is the right shape for
a load-balanced, autoscaled service where any task may handle any request.

Every response carries the same fields eval/trajectory_eval.py and
eval/online_eval.py already know how to score: `trajectory` (the ordered
grounding/tool steps this turn actually took) and `trace_id` (so an external
scorer can attach LLM-judge / trajectory scores back onto the exact LangFuse
trace agent/harness.py already created for this request).
"""
import os
import sys

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import uuid

from agent.hitl import ApprovalAlreadyDecidedError, ApprovalExpiredError, StaleStateError
from agent.hitl_store import HITLStore
from agent.harness import SupportHarness, resume_after_approval
from agent.tracing import langfuse

load_dotenv()

app = FastAPI(title="ecommerce-support-agent")


class ChatRequest(BaseModel):
    customer_id: str
    message: str


class ChatResponse(BaseModel):
    response: str
    trajectory: list[str]
    trace_id: str | None = None


class ApprovalDecideRequest(BaseModel):
    status: str  # "approved" | "rejected"
    decided_by: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest):
    if not body.message:
        raise HTTPException(status_code=400, detail="missing 'message'")

    try:
        async with SupportHarness(body.customer_id, ticket_id=uuid.uuid4().hex) as harness:
            response_text = await harness.handle_turn(body.message)
            trajectory = harness.trajectory()
            trace_id = harness.last_trace_id
    except ValueError as exc:  # unknown customer_id
        raise HTTPException(status_code=400, detail=str(exc))

    # Short-lived request/response cycle -- flush now so the trace shows up
    # in the dashboard immediately instead of waiting on the SDK's own batch
    # timer, same reasoning as agent-cicd-demo's app/server.py.
    langfuse.flush()
    return ChatResponse(response=response_text, trajectory=trajectory, trace_id=trace_id)


@app.get("/approvals")
def list_approvals():
    """Enough for a human-reviewer flow to be testable end to end -- no
    polished UI, just the pending queue a real reviewer would work from."""
    store = HITLStore()
    return {"pending": [a.model_dump() for a in store.list_pending()]}


@app.post("/approvals/{approval_id}/decide")
async def decide_approval(approval_id: str, body: ApprovalDecideRequest):
    try:
        result = await resume_after_approval(approval_id, body.status, body.decided_by)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except (ApprovalAlreadyDecidedError, ApprovalExpiredError) as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except StaleStateError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return result


if __name__ == "__main__":
    import uvicorn

    if not os.environ.get("GROQ_API_KEY"):
        print(
            "GROQ_API_KEY is not set. Copy .env.example to .env and put your "
            "Groq API key in it (https://console.groq.com/keys), then run this again."
        )
        sys.exit(1)

    uvicorn.run(app, host="0.0.0.0", port=8080)
