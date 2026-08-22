"""MCP server exposing two read-only e-commerce tools.

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
ever dispatching to this server. This server additionally binds itself to a
single customer_id for its whole process lifetime (passed via the
SESSION_CUSTOMER_ID environment variable when the harness spawns it), so even
a direct call to this process outside the harness cannot cross customer
boundaries.
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

SESSION_CUSTOMER_ID = os.environ.get("SESSION_CUSTOMER_ID")

_ORDER_ID_RE = re.compile(r"^ORD\d+$")
_CUSTOMER_ID_RE = re.compile(r"^CUST\d+$")

mcp = FastMCP("ecommerce-support")


@mcp.tool()
def lookup_order(order_id: str) -> dict:
    """Look up an order by its order ID.

    Returns the order's status, items, and delivery date. Only orders
    belonging to the session's authenticated customer may be looked up;
    any other order_id raises a ToolError.
    """
    if not _ORDER_ID_RE.match(order_id or ""):
        raise ToolError(f"Malformed order_id: {order_id!r} (expected e.g. 'ORD1001').")

    order = ORDERS.get(order_id)
    if order is None:
        raise ToolError(f"No such order: {order_id}")

    if SESSION_CUSTOMER_ID is None:
        raise ToolError(
            "PermissionError: this server is not bound to an authenticated "
            "session (SESSION_CUSTOMER_ID unset). Refusing to serve any "
            "request rather than skip the ownership check."
        )
    if order["customer_id"] != SESSION_CUSTOMER_ID:
        raise ToolError(
            f"PermissionError: order {order_id} does not belong to the "
            f"authenticated customer {SESSION_CUSTOMER_ID}. Request rejected."
        )
    return order


@mcp.tool()
def issue_refund(order_id: str, amount_usd: float) -> dict:
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

    if SESSION_CUSTOMER_ID is None:
        raise ToolError(
            "PermissionError: this server is not bound to an authenticated "
            "session (SESSION_CUSTOMER_ID unset). Refusing to serve any "
            "request rather than skip the ownership check."
        )
    if order["customer_id"] != SESSION_CUSTOMER_ID:
        raise ToolError(
            f"PermissionError: order {order_id} does not belong to the "
            f"authenticated customer {SESSION_CUSTOMER_ID}. Request rejected."
        )
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
    """Look up account standing and order history for a customer ID.

    Returns standing (active/flagged/suspended), subscription, and order
    history. Only the session's authenticated customer_id may be queried;
    any other customer_id raises a ToolError.
    """
    if not _CUSTOMER_ID_RE.match(customer_id or ""):
        raise ToolError(f"Malformed customer_id: {customer_id!r} (expected e.g. 'CUST001').")

    if SESSION_CUSTOMER_ID is None:
        raise ToolError(
            "PermissionError: this server is not bound to an authenticated "
            "session (SESSION_CUSTOMER_ID unset). Refusing to serve any "
            "request rather than skip the ownership check."
        )
    if customer_id != SESSION_CUSTOMER_ID:
        raise ToolError(
            f"PermissionError: customer {customer_id} does not match the "
            f"authenticated session {SESSION_CUSTOMER_ID}. Request rejected."
        )

    account = ACCOUNTS.get(customer_id)
    if account is None:
        raise ToolError(f"No such customer: {customer_id}")
    return account


if __name__ == "__main__":
    mcp.run(transport="stdio")
