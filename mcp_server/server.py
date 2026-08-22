"""MCP server exposing three e-commerce tools (two read-only, one write).

Schema validation is handled by FastMCP itself: it derives a JSON schema from
each function's type hints, and a call with a missing/mismatched field never
reaches the function body -- the MCP layer rejects it before our code runs.

Both failure kinds (not-found, permission-denied) are raised as ToolError,
which MCP reports back to the client as a proper tool error (isError=True),
not folded into an ordinary-looking successful result the caller has to
inspect for an "error" key by convention.

Permission scoping here is defense-in-depth, not the primary boundary. The
primary boundary lives in the harness (agent/harness.py), which checks a
proposed tool call against the ticket's authenticated customer_id *before*
ever dispatching to this server.

customer_id is a real, explicit parameter on every tool (Assignment 3 --
Gateway migration), not a module-level SESSION_CUSTOMER_ID env var read once
at process startup. That env-var design only worked because every ticket got
its own fresh stdio subprocess (agent/mcp_client.py's original design) --
once this server runs as a single persistent process serving many customers
concurrently (behind AgentCore Gateway, agent/mcp_client_gateway.py), a
module-level global would race across concurrent requests from different
customers, a real correctness/security bug, not a style preference. The
calling client is responsible for supplying the REAL authenticated
customer_id here, never a value taken from the model's own tool-call
arguments -- see agent/mcp_client_gateway.py's call_tool() for where that's
enforced (the client overwrites customer_id unconditionally before sending,
regardless of what a tool call otherwise contains).

Transport: stdio by default (unchanged, for the original per-ticket-subprocess
path, agent/mcp_client.py). MCP_TRANSPORT=streamable-http switches to HTTP,
required for AgentCore Runtime's MCP protocol contract (port 8000, path
/mcp) when this server is deployed as its own Runtime app and fronted by
AgentCore Gateway.
"""
import json
import os
import re
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ORDERS = json.loads((DATA_DIR / "orders.json").read_text(encoding="utf-8"))
ACCOUNTS = json.loads((DATA_DIR / "accounts.json").read_text(encoding="utf-8"))

_ORDER_ID_RE = re.compile(r"^ORD\d+$")
_CUSTOMER_ID_RE = re.compile(r"^CUST\d+$")

mcp = FastMCP("ecommerce-support", port=8000)


def _check_owns(resource_customer_id: str, customer_id: str, what: str, ref: str) -> None:
    if not _CUSTOMER_ID_RE.match(customer_id or ""):
        raise ToolError(f"Malformed customer_id: {customer_id!r} (expected e.g. 'CUST001').")
    if resource_customer_id != customer_id:
        raise ToolError(f"PermissionError: {what} {ref} does not belong to customer {customer_id}. Request rejected.")


@mcp.tool()
def lookup_order(order_id: str, customer_id: str) -> dict:
    """Look up an order by its order ID.

    Returns the order's status, items, and delivery date. Only orders
    belonging to customer_id may be looked up; any other order_id raises a
    ToolError. customer_id must be the real, authenticated caller -- see
    this module's docstring on why that value is never taken from the
    model's own tool-call arguments by the client that calls this tool.
    """
    if not _ORDER_ID_RE.match(order_id or ""):
        raise ToolError(f"Malformed order_id: {order_id!r} (expected e.g. 'ORD1001').")

    order = ORDERS.get(order_id)
    if order is None:
        raise ToolError(f"No such order: {order_id}")

    _check_owns(order["customer_id"], customer_id, "order", order_id)
    return order


@mcp.tool()
def issue_refund(order_id: str, amount_usd: float, customer_id: str) -> dict:
    """Issue a refund for an order. Only called for orders already cleared
    by agent/policy_boundary.py's evaluate_refund_policy() -- either
    auto-approved under the approval threshold, or approved by a human via
    the HITL gate (agent/hitl.py) after re-validation. This tool itself
    still re-checks ownership and amount as defense-in-depth, the same
    multi-layer discipline as lookup_order/check_account_status, never
    trusting a single upstream check alone.

    Mutates the in-process ORDERS dict only (marks refund_issued=True,
    refunded_amount=amount_usd) -- not persisted back to orders.json. This
    is a mock backend for a demo agent, same as lookup_order/
    check_account_status never persisting anything either; a real backend
    would call an actual payments API here.
    """
    if not _ORDER_ID_RE.match(order_id or ""):
        raise ToolError(f"Malformed order_id: {order_id!r} (expected e.g. 'ORD1001').")

    order = ORDERS.get(order_id)
    if order is None:
        raise ToolError(f"No such order: {order_id}")

    _check_owns(order["customer_id"], customer_id, "order", order_id)
    if order.get("refund_issued"):
        raise ToolError(f"Order {order_id} already has a refund on record; refusing to issue a second one.")
    if abs(amount_usd - order["order_total"]) > 0.01:
        raise ToolError(
            f"Refund amount ${amount_usd:.2f} does not match order total on record "
            f"(${order['order_total']:.2f}); refusing to issue."
        )

    order["refund_issued"] = True
    order["refunded_amount"] = amount_usd
    return {"order_id": order_id, "refund_issued": True, "refunded_amount": amount_usd}


@mcp.tool()
def check_account_status(customer_id: str) -> dict:
    """Look up account standing and order history for the authenticated
    customer_id.

    Returns standing (active/flagged/suspended), subscription, and order
    history.
    """
    if not _CUSTOMER_ID_RE.match(customer_id or ""):
        raise ToolError(f"Malformed customer_id: {customer_id!r} (expected e.g. 'CUST001').")

    account = ACCOUNTS.get(customer_id)
    if account is None:
        raise ToolError(f"No such customer: {customer_id}")
    return account


if __name__ == "__main__":
    transport = os.environ.get("MCP_TRANSPORT", "stdio")
    mcp.run(transport=transport)
