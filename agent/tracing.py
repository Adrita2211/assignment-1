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


def safe_span_payload(payload: dict | list):
    """PII enforcement point #3 (see agent/pii.py's module docstring): a
    trace store is a data store, full stop -- redact every string value in
    a span's input/output dict before it reaches update_current_span/
    update_current_generation, same as any other PII exit point. Imported
    lazily (not at module top level) to avoid Presidio/spaCy's real
    startup cost for every process that imports agent.tracing but never
    actually traces PII-bearing content (e.g. a pure eval-gate run)."""
    from agent.pii import redact_value

    return redact_value(payload)


__all__ = ["langfuse", "observe", "safe_span_payload"]
