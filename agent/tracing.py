"""LangFuse client setup, shared by agent/harness.py, server.py, and eval/*.

Same pattern as agent-cicd-demo's app/agent.py: no tracing config at all in
some contexts (offline eval, CI eval-gate, a local run with no LangFuse key)
is expected, not an error. Two real SDK noise sources show up whenever no key
is present, both silenced here, both left untouched whenever a key IS
present -- these signals catch real tracing bugs (a wrong host, expired
key), so suppressing them unconditionally would hide genuine failures:

  1. Langfuse.__init__() logs "Authentication error... Client will be
     disabled" synchronously, before get_client() even returns -- has to be
     silenced BEFORE the call.
  2. The SDK still spins up a background exporter that periodically retries
     a doomed network call on its own timer -- logger-level suppression
     after the fact loses that fight. Disabling tracing at the SDK level via
     LANGFUSE_TRACING_ENABLED fixes the actual cause instead of hiding it.
"""
import logging
import os

from langfuse import get_client, observe

if not os.environ.get("LANGFUSE_PUBLIC_KEY"):
    logging.getLogger("langfuse").setLevel(logging.CRITICAL)
    os.environ["LANGFUSE_TRACING_ENABLED"] = "false"

langfuse = get_client()

__all__ = ["langfuse", "observe"]
