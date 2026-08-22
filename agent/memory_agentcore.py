"""AgentCore Memory-backed short-term conversation memory -- the real
managed-session-state implementation, replacing agent/memory.py's
ShortTermMemory (an in-process, never-persisted-anywhere buffer) once
deployed to AgentCore Runtime.

Real gap this closes: agent/memory.py's ShortTermMemory only lives as long
as one SupportHarness instance -- and SupportHarness is constructed fresh
per turn (agentcore_app.py's invoke(), server.py's /chat), so multi-turn
conversation memory has never actually persisted across separate
invocations in this project before this module existed, regardless of
backend. AgentCore Memory (`bedrock_agentcore.memory.MemoryClient`,
confirmed against the real installed SDK, not assumed from a blog post --
create_event/get_last_k_turns signatures verified via inspect.signature())
is what makes that persistence real, keyed on (memory_id, actor_id=
customer_id, session_id=ticket_id) -- so persistence across turns of one
support ticket requires the caller to reuse the same ticket_id across
invocations, exactly as agentcore_app.py's payload already allows
(payload.get("ticket_id"), not always a fresh uuid).

Same public shape as ShortTermMemory (add()/as_list()) so
agent/harness.py's SupportHarness needs only a construction-site swap, not
a redesign -- plus one new method, persist_turn(), that handle_turn() calls
once a turn completes, since AgentCore Memory has no equivalent of
ShortTermMemory's "just mutate self.messages" pattern; writes are explicit
events, not an implicit buffer.
"""
from __future__ import annotations

import os


class ShortTermMemoryAgentCore:
    def __init__(self, customer_id: str, ticket_id: str, memory_id: str | None = None, region: str | None = None):
        self.customer_id = customer_id
        self.ticket_id = ticket_id
        self.memory_id = memory_id or os.environ.get("BEDROCK_AGENTCORE_MEMORY_ID")
        self._region = region or os.environ.get("AWS_REGION", "us-east-1")
        self._client = None
        self.messages: list[dict] = []
        self._loaded = False

    @property
    def client(self):
        if self._client is None:
            from bedrock_agentcore.memory import MemoryClient

            self._client = MemoryClient(region_name=self._region)
        return self._client

    def _load_prior_turns(self) -> list[dict]:
        """Real AgentCore Memory turns come back as List[List[Dict]] (one
        inner list per stored turn, each dict shaped like {"role":
        "USER"/"ASSISTANT", "content": {"text": ...}} per the SDK's own
        event schema) -- flattened here into this project's plain
        OpenAI-shaped {"role": "user"/"assistant", "content": str} messages
        so nothing downstream (agent/harness.py's decide_node, the
        provider's message translation) needs to know the source."""
        if self.memory_id is None:
            return []
        turns = self.client.get_last_k_turns(
            memory_id=self.memory_id, actor_id=self.customer_id, session_id=self.ticket_id, k=5,
        )
        flattened: list[dict] = []
        for turn in turns:
            for item in turn:
                role = item.get("role", "").lower()
                text = item.get("content", {}).get("text", "")
                if role in ("user", "assistant") and text:
                    flattened.append({"role": role, "content": text})
        return flattened

    def add(self, message: dict):
        self.messages.append(message)

    def as_list(self) -> list[dict]:
        if not self._loaded:
            self.messages = self._load_prior_turns() + self.messages
            self._loaded = True
        return self.messages

    def persist_turn(self, user_text: str, assistant_text: str) -> None:
        if self.memory_id is None:
            return
        self.client.create_event(
            memory_id=self.memory_id,
            actor_id=self.customer_id,
            session_id=self.ticket_id,
            messages=[(user_text, "USER"), (assistant_text, "ASSISTANT")],
        )


class LongTermMemoryAgentCore:
    """AgentCore Memory long-term extraction strategies -- the real
    version of agent/memory.py's LongTermMemory, which is a static,
    hand-written JSON lookup (data/ticket_history.json) that never grows
    or updates from real conversations. That's not actually memory in the
    sense the term usually means; it's a fixed seed dataset.

    This class is backed by three real extraction strategies added to the
    same AgentCore Memory resource ShortTermMemoryAgentCore already uses
    (SEMANTIC, SUMMARY, USER_PREFERENCE -- see the strategy-creation calls
    in this project's history/README for the real namespaces used).
    AgentCore runs an LLM over stored conversation events in the
    background to extract durable facts/summaries/preferences -- a plain
    database never does this on its own; it only stores what's explicitly
    written into it. retrieve_memories() is a semantic search over
    whatever's actually been extracted so far, which is why this class's
    result can legitimately be empty right after a conversation happens --
    extraction is asynchronous, not instant, unlike ShortTermMemoryAgentCore's
    create_event/get_last_k_turns round trip.

    Same public method as LongTermMemory (get_history_summary) so
    agent/harness.py's calling code needs no redesign."""

    def __init__(self, memory_id: str | None = None, region: str | None = None):
        self.memory_id = memory_id or os.environ.get("BEDROCK_AGENTCORE_MEMORY_ID")
        self._region = region or os.environ.get("AWS_REGION", "us-east-1")
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from bedrock_agentcore.memory import MemoryClient

            self._client = MemoryClient(region_name=self._region)
        return self._client

    def get_history_summary(self, customer_id: str) -> str:
        if self.memory_id is None:
            return "No prior support tickets on file for this customer."

        facts = self.client.retrieve_memories(
            memory_id=self.memory_id, namespace=f"ecommerce_facts/{customer_id}",
            actor_id=customer_id, query="past support tickets and preferences for this customer", top_k=5,
        )
        prefs = self.client.retrieve_memories(
            memory_id=self.memory_id, namespace=f"ecommerce_prefs/{customer_id}",
            actor_id=customer_id, query="customer preferences", top_k=3,
        )
        lines = []
        for item in facts + prefs:
            text = item.get("content", {}).get("text") if isinstance(item.get("content"), dict) else item.get("text")
            if text:
                lines.append(f"- {text}")
        if not lines:
            return "No prior support tickets on file for this customer."
        return "Prior ticket history for this customer (AgentCore-extracted):\n" + "\n".join(lines)
