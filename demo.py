"""Scripted demo: runs several ticket scenarios end to end, plus the
"craft" demonstrations the assignment explicitly asks to see live:

  1. A harness-level permission rejection (customer asks about an order
     that isn't theirs -- rejected in agent/harness.py:
     _validate_and_check_permission, BEFORE any MCP call is dispatched).
  2. An MCP schema-layer rejection (missing field, wrong type) sent
     directly to the MCP client -- rejected by FastMCP's generated schema,
     not a manual if-check.
  3. A hybrid-RAG synonym rescue: a question that shares ZERO words with
     the doc that actually answers it -- pure BM25 would find nothing, the
     vector (semantic) half of the fusion is what rescues it.
  4. The honest-gap case: a plausible question (customs fees) that nothing
     in the policy corpus covers, so the agent says so instead of guessing.

Run with: python demo.py
"""
import asyncio
import os
import sys

from dotenv import load_dotenv

from agent.harness import SupportHarness
from agent.mcp_client import MCPToolClient
from agent.rag import HybridPolicyRetriever


def _banner(title: str):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


SCENARIOS = [
    ("CUST002", "order_status", "Hi, where is my order ORD1002? Has it shipped yet?"),
    ("CUST001", "delivery_issue", "My monitor order ORD1003 is really late. What can you do about it?"),
    ("CUST004", "refund_request", "My desk lamp from ORD1005 arrived damaged. Can I get a refund?"),
    (
        "CUST005",
        "subscription_account",
        "I heard my account got suspended -- how do I appeal that, and can I still cancel my Plus subscription while it's under review?",
    ),
    ("CUST002", "out_of_scope_permission_test", "Can you check order ORD1005 for me and tell me its status?"),
    ("CUST001", "rag_honest_gap", "Do you cover customs fees for international orders that get held at customs?"),
    (
        "CUST001",
        "rag_hybrid_synonym_rescue",
        "If I no longer want the thing I bought and haven't opened it yet, will you give me my money back?",
    ),
]


async def run_conversation_scenarios():
    for customer_id, label, message in SCENARIOS:
        _banner(f"Scenario [{label}] as {customer_id}")
        print(f"user: {message}\n")
        async with SupportHarness(customer_id) as harness:
            reply = await harness.handle_turn(message)
        print(f"\nagent: {reply}")


def run_rag_fusion_breakdown():
    _banner("Craft demo: hybrid RAG score breakdown (bm25 / vector / fused)")
    retriever = HybridPolicyRetriever("policies")
    query = "If I no longer want the thing I bought and haven't opened it yet, will you give me my money back?"
    print(f"Query: {query}")
    print("(Zero shared vocabulary with 'return_window.md' -- lexical/BM25 alone would score it 0.)\n")
    for doc, breakdown in retriever.retrieve(query, top_k=3):
        print(f"  {doc['id']:28s} bm25={breakdown['bm25']:.3f}  vector={breakdown['vector']:.3f}  fused={breakdown['fused']:.3f}")


async def run_harness_permission_demo():
    _banner("Craft demo: harness rejects a cross-customer tool call BEFORE MCP dispatch")
    async with SupportHarness("CUST002") as harness:
        allowed, category, parsed = harness._validate_and_check_permission(
            "lookup_order", {"order_id": "ORD1005"}  # belongs to CUST004, not CUST002
        )
        print(f"CUST002 asks about ORD1005 (belongs to CUST004): allowed={allowed} category={category!r}")
        print("This never reaches agent/mcp_client.py -- rejected in the harness itself.")


async def run_malformed_call_demo():
    _banner("Craft demo: malformed tool call rejected by the MCP schema layer")
    async with MCPToolClient("CUST001") as client:
        print("Calling lookup_order with NO order_id argument (missing required field)...")
        result = await client.call_tool("lookup_order", {})
        print("Result:", result)

        print("\nCalling lookup_order with order_id as a list instead of a string...")
        result = await client.call_tool("lookup_order", {"order_id": ["ORD1002"]})
        print("Result:", result)

    _banner("Craft demo: harness-level schema rejection (extra/unexpected field)")
    async with SupportHarness("CUST001") as harness:
        allowed, category, parsed = harness._validate_and_check_permission(
            "lookup_order", {"order_id": "ORD1003", "admin_override": True}
        )
        print(f"lookup_order with an unexpected 'admin_override' field: allowed={allowed} category={category!r}")
        print("Rejected by the pydantic model (extra='forbid') before it ever reaches MCP.")


async def main():
    load_dotenv()
    if not os.environ.get("GROQ_API_KEY"):
        print(
            "GROQ_API_KEY is not set. Copy .env.example to .env and put your "
            "Groq API key in it (https://console.groq.com/keys), then run this again."
        )
        sys.exit(1)
    await run_conversation_scenarios()
    run_rag_fusion_breakdown()
    await run_harness_permission_demo()
    await run_malformed_call_demo()
    _banner("Demo complete")


if __name__ == "__main__":
    asyncio.run(main())
