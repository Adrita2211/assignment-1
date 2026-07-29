"""Async MCP client: spawns mcp_server/server.py as a stdio subprocess, one
session per ticket, bound to that ticket's customer_id.

This is NOT where permission scoping is primarily enforced -- see
agent/harness.py's _check_permission, which runs before this client is ever
called. This client exists to (a) actually perform the MCP round trip and
(b) surface schema-validation and defense-in-depth rejections that happen
inside the MCP layer itself, so those are visible as real rejections rather
than something the harness fabricates.
"""
import json
import os
import sys
from contextlib import AsyncExitStack
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER_SCRIPT = Path(__file__).resolve().parent.parent / "mcp_server" / "server.py"


class MCPToolClient:
    def __init__(self, customer_id: str):
        self.customer_id = customer_id
        self._stack = AsyncExitStack()
        self.session: ClientSession | None = None

    async def __aenter__(self) -> "MCPToolClient":
        server_params = StdioServerParameters(
            command=sys.executable,
            args=[str(SERVER_SCRIPT)],
            env={**os.environ, "SESSION_CUSTOMER_ID": self.customer_id},
        )
        read, write = await self._stack.enter_async_context(stdio_client(server_params))
        self.session = await self._stack.enter_async_context(ClientSession(read, write))
        await self.session.initialize()
        return self

    async def __aexit__(self, *exc_info):
        await self._stack.aclose()

    async def call_tool(self, name: str, arguments: dict) -> dict:
        """Call an MCP tool and return a plain dict result.

        If the MCP layer rejects the call (bad schema, or the server's own
        SESSION_CUSTOMER_ID check), that shows up as {"error": ...} rather
        than raising, so callers can log/handle it uniformly.
        """
        try:
            result = await self.session.call_tool(name, arguments)
        except Exception as exc:  # MCP schema/validation errors surface here
            return {"error": "mcp_layer_rejected", "detail": str(exc)}

        text_parts = [block.text for block in result.content if getattr(block, "text", None)]
        payload = "\n".join(text_parts) if text_parts else "{}"
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            parsed = {"raw": payload}

        if result.isError:
            return {"error": "mcp_layer_rejected", "detail": parsed}
        return parsed
