"""The harness loop, as a LangGraph StateGraph.

This is the one file where the "LLM proposes, harness decides" pattern has
to be real. The graph shape is:

    classify -> retrieve -> memory -> decide
                                          |
                          (conditional edge: any tool calls proposed?)
                                          |
                       no tool calls -> respond -> END
                                          |
                                     tool calls
                                          |
                                       validate  <-- permission checks live
                                          |          HERE, structurally
                                          v          between decide and act,
                                        act          never inside the
                                          |          decide/LLM node.
                     (conditional edge: iteration cap hit?)
                            |                    |
                       decide (loop)         respond -> END

`validate` is the node the assignment calls "a conditional edge between
decide and act": LangGraph's routing functions (the things passed to
`add_conditional_edges`) can only choose the next node, they can't also
produce state updates, so the actual permission-check *logic*
(`SupportHarness._validate_and_check_permission`) has to live in a node
rather than in the router callback itself. `validate` is that node -- it
sits on every path from `decide` to `act`, runs before `act` ever touches
`agent/mcp_client.py`, and is a completely separate node from `decide` (the
only node that talks to the LLM). A rejected call never reaches
`act`'s MCP dispatch at all -- `act` still runs (it has to feed a rejection
`tool_result` back to the model), but for a rejected call it never calls
`self.mcp_client.call_tool(...)`.

Security layers, in the order a proposed call actually passes through
`validate` (see `SupportHarness._validate_and_check_permission`):
  1. Tool allowlist -- the tool name must be one of the two known tools.
  2. Schema/shape validation -- a pydantic model with extra="forbid" rejects
     missing fields, wrong types, or unexpected extra keys, independent of
     (and before) whatever the MCP server would also reject.
  3. ID format validation -- order_id/customer_id must match the expected
     ID pattern (ORD\\d+ / CUST\\d+), catching garbage input cheaply.
  4. Ownership check -- the resolved order/customer must belong to THIS
     ticket's authenticated customer_id.
Only a call that clears all four is dispatched to MCP inside `act`. Every
decision (allowed or rejected, and why) is appended to self.audit_log.

A per-turn tool-call round-trip cap (MAX_TOOL_ITERATIONS) stops a
misbehaving or adversarial model from looping tool calls indefinitely --
enforced by the conditional edge after `act`.
"""
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, ValidationError

from agent.classify import classify_ticket
from agent.memory import LongTermMemory, ShortTermMemory
from agent.mcp_client import MCPToolClient
from agent.provider import GroqProvider
from agent.rag import HybridPolicyRetriever

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MAX_TOOL_ITERATIONS = 6

_ORDER_ID_RE = re.compile(r"^ORD\d+$")
_CUSTOMER_ID_RE = re.compile(r"^CUST\d+$")


class LookupOrderArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    order_id: str

    @property
    def is_well_formed(self) -> bool:
        return bool(_ORDER_ID_RE.match(self.order_id))


class CheckAccountStatusArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    customer_id: str

    @property
    def is_well_formed(self) -> bool:
        return bool(_CUSTOMER_ID_RE.match(self.customer_id))


_ARG_MODELS: dict[str, type[BaseModel]] = {
    "lookup_order": LookupOrderArgs,
    "check_account_status": CheckAccountStatusArgs,
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_order",
            "description": (
                "Look up an order by its order ID. Returns status, items, and "
                "delivery date. Only works for orders belonging to the current "
                "customer."
            ),
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string", "description": "e.g. ORD1002"}},
                "required": ["order_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_account_status",
            "description": (
                "Look up account standing (active/flagged/suspended) and order "
                "history for a customer ID. Only works for the current "
                "authenticated customer."
            ),
            "parameters": {
                "type": "object",
                "properties": {"customer_id": {"type": "string", "description": "e.g. CUST002"}},
                "required": ["customer_id"],
                "additionalProperties": False,
            },
        },
    },
]

SYSTEM_TEMPLATE = """You are a customer support agent for an e-commerce store.

The customer you are talking to is authenticated as: {customer_id}
This ticket has been classified as: {ticket_type}

Relevant policy context (retrieved for this message):
{policy_context}

{history_summary}

Rules:
- Only use lookup_order / check_account_status for THIS customer's own data.
  Never invent order or account data -- always call the tool to get it.
- If the retrieved policy context above says nothing is relevant, tell the
  customer plainly that this isn't covered by policy you have access to and
  that it will be escalated -- do not guess at a policy or invent numbers.
- Do not offer, promise, or process a refund -- refund issuance is not
  available to you. You may explain refund *eligibility* per policy and say
  the request will be escalated.
- Be concise and concrete (cite the actual policy numbers you were given,
  and only numbers you were given).
"""


def build_system_prompt(ticket_type: str, customer_id: str, policy_context: str, history_summary: str) -> str:
    return SYSTEM_TEMPLATE.format(
        customer_id=customer_id,
        ticket_type=ticket_type,
        policy_context=policy_context,
        history_summary=history_summary,
    )


@lru_cache(maxsize=1)
def _shared_retriever() -> HybridPolicyRetriever:
    """The policy corpus never changes between tickets, so build the BM25
    index and load the embedding model exactly once per process and reuse it
    across every SupportHarness session, instead of paying that startup cost
    (most visibly, reloading the embedding model) on every single ticket."""
    return HybridPolicyRetriever(Path(__file__).resolve().parent.parent / "policies")


# --- Graph state ------------------------------------------------------------
# Everything a node needs to read or write for a single handle_turn() call.
# Instance-specific dependencies (provider, retriever, mcp_client, audit_log,
# ...) are NOT part of this state -- they're reached via
# config["configurable"]["harness"], so the compiled graph itself stays a
# stateless singleton shared across every SupportHarness session, the same
# way _shared_retriever() is shared.

class TurnState(TypedDict):
    messages: list[dict]              # the full short-term buffer, OpenAI chat format
    customer_id: str
    ticket_type: str
    policy_context: str
    history_summary: str
    pending_tool_calls: list[dict]    # tool calls proposed by the last `decide` call
    validated_calls: list[dict]       # `validate`'s output: allow/reject decision per call
    iteration_count: int
    final_text: Optional[str]


def _get_harness(config: RunnableConfig) -> "SupportHarness":
    return config["configurable"]["harness"]


def _latest_user_text(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m["role"] == "user":
            return m["content"]
    return ""


_RETRIEVAL_CONTEXT_WINDOW = 4  # last N user/assistant turns, not just this message


def _retrieval_query(messages: list[dict]) -> str:
    """A vague follow-up ("tell me the policy on this") has no topical
    content by itself -- its subject lives in the preceding turn(s). Build
    the retrieval query from the last few user/assistant turns (skipping
    tool-call/tool-result messages, which aren't natural-language text)
    instead of just the single latest message, so a short follow-up can
    still retrieve the right doc."""
    recent = [
        m for m in messages
        if m["role"] in ("user", "assistant") and isinstance(m.get("content"), str) and m["content"]
    ]
    return "\n".join(m["content"] for m in recent[-_RETRIEVAL_CONTEXT_WINDOW:])


# --- Nodes -------------------------------------------------------------------

async def classify_node(state: TurnState, config: RunnableConfig) -> dict:
    ticket_type = classify_ticket(_latest_user_text(state["messages"]))
    print(f"[CLASSIFY] ticket_type={ticket_type}")
    return {"ticket_type": ticket_type}


async def retrieve_node(state: TurnState, config: RunnableConfig) -> dict:
    """Try the current message alone first -- that's the strongest, least
    noisy signal, and is what every accuracy test was tuned against. Only
    fall back to a multi-turn context window if the message alone retrieves
    nothing: concatenating several turns unconditionally was tried and
    reverted, because a verbose prior turn's vocabulary (e.g. an earlier
    escalation reply full of unrelated words) can outweigh and misrank a
    perfectly well-formed new question that would have retrieved correctly
    on its own."""
    harness = _get_harness(config)
    current_text = _latest_user_text(state["messages"])
    hits = harness.retriever.retrieve(current_text, top_k=2)
    used_context = False

    if not hits:
        context_query = _retrieval_query(state["messages"])
        if context_query != current_text:
            hits = harness.retriever.retrieve(context_query, top_k=2)
            used_context = bool(hits)

    if hits:
        policy_context = "\n\n".join(f"[{doc['id']}] {doc['text']}" for doc, _ in hits)
        tag = " (via conversation context fallback)" if used_context else ""
        print(f"[RAG] retrieved{tag}: {[(doc['id'], score) for doc, score in hits]}")
    else:
        policy_context = "(No policy document in the knowledge base is relevant to this question.)"
        print("[RAG] no relevant chunk found above threshold -- honest gap")
    return {"policy_context": policy_context}


async def memory_node(state: TurnState, config: RunnableConfig) -> dict:
    harness = _get_harness(config)
    history_summary = harness.long_term.get_history_summary(harness.customer_id)
    print(f"[MEMORY] long-term (customer_id={harness.customer_id}): {history_summary}")
    print(f"[MEMORY] short-term buffer size before this turn: {len(state['messages']) - 1} messages")
    return {"history_summary": history_summary}


async def decide_node(state: TurnState, config: RunnableConfig) -> dict:
    """The only node that talks to the LLM. It proposes tool calls; it does
    NOT decide whether they're allowed to run -- that's `validate`'s job."""
    harness = _get_harness(config)
    system_msg = {
        "role": "system",
        "content": build_system_prompt(
            state["ticket_type"], state["customer_id"], state["policy_context"], state["history_summary"]
        ),
    }
    result = harness.provider.call([system_msg] + state["messages"], tools=TOOLS)

    if result["tool_calls"]:
        assistant_msg = {
            "role": "assistant",
            "content": result["text"],
            "tool_calls": [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])},
                }
                for tc in result["tool_calls"]
            ],
        }
        return {
            "messages": state["messages"] + [assistant_msg],
            "pending_tool_calls": result["tool_calls"],
        }

    final_text = result["text"] or ""
    return {
        "messages": state["messages"] + [{"role": "assistant", "content": final_text}],
        "pending_tool_calls": [],
        "final_text": final_text,
    }


async def validate_node(state: TurnState, config: RunnableConfig) -> dict:
    """THE permission-check boundary. Sits structurally between `decide` and
    `act`: every tool call proposed by `decide` passes through here, and
    `act` only ever dispatches to MCP for calls this node marked allowed."""
    harness = _get_harness(config)
    validated = []
    for tc in state["pending_tool_calls"]:
        allowed, category, parsed_args = harness._validate_and_check_permission(tc["name"], tc["arguments"])
        reason = None if allowed else harness._reason_text(tc["name"], tc["arguments"], category)
        if allowed:
            print(f"[HARNESS] ALLOWED {tc['name']}({parsed_args})")
        else:
            print(f"[HARNESS] REJECTED ({category}) {tc['name']}({tc['arguments']}) -> {reason}")
        validated.append({
            "tool_call": tc, "allowed": allowed, "category": category,
            "parsed_args": parsed_args, "reason": reason,
        })
    return {"validated_calls": validated}


async def act_node(state: TurnState, config: RunnableConfig) -> dict:
    """Dispatches to MCP -- but ONLY for calls `validate` already marked
    allowed. A rejected call never reaches self.mcp_client.call_tool(...);
    it gets a synthetic rejection tool_result instead, same as an allowed
    call gets a real one, so the model always receives a response either way."""
    harness = _get_harness(config)
    new_messages = list(state["messages"])

    for v in state["validated_calls"]:
        tc = v["tool_call"]
        if v["allowed"]:
            harness.audit_log.append({
                "tool": tc["name"], "args": v["parsed_args"], "decision": "allowed", "category": "allowed",
            })
            tool_result = await harness.mcp_client.call_tool(tc["name"], v["parsed_args"])
            content = json.dumps(tool_result)
        else:
            harness.audit_log.append({
                "tool": tc["name"], "args": tc["arguments"], "decision": "rejected",
                "category": v["category"], "reason": v["reason"],
            })
            content = json.dumps({"error": "rejected_by_harness", "category": v["category"], "reason": v["reason"]})
        new_messages.append({"role": "tool", "tool_call_id": tc["id"], "content": content})

    return {
        "messages": new_messages,
        "iteration_count": state["iteration_count"] + 1,
        "pending_tool_calls": [],
        "validated_calls": [],
    }


async def respond_node(state: TurnState, config: RunnableConfig) -> dict:
    """No-op if `decide` already produced a final answer. Only does real
    work on the safety-valve path: MAX_TOOL_ITERATIONS was hit without the
    model ever settling on a plain-text answer."""
    if state.get("final_text") is not None:
        return {}
    print(f"[HARNESS] MAX_TOOL_ITERATIONS ({MAX_TOOL_ITERATIONS}) reached; escalating.")
    fallback = (
        "I wasn't able to finish resolving this in a bounded number of "
        "tool calls, so I'm escalating this ticket to a human agent."
    )
    return {
        "messages": state["messages"] + [{"role": "assistant", "content": fallback}],
        "final_text": fallback,
    }


# --- Conditional edges ---------------------------------------------------

def _route_after_decide(state: TurnState) -> str:
    return "validate" if state["pending_tool_calls"] else "respond"


def _route_after_act(state: TurnState) -> str:
    return "respond" if state["iteration_count"] >= MAX_TOOL_ITERATIONS else "decide"


@lru_cache(maxsize=1)
def _compiled_graph():
    """Built once per process (like _shared_retriever) -- the graph
    structure is identical for every ticket; only the state passed to
    ainvoke() differs per turn."""
    graph = StateGraph(TurnState)
    graph.add_node("classify", classify_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("memory", memory_node)
    graph.add_node("decide", decide_node)
    graph.add_node("validate", validate_node)
    graph.add_node("act", act_node)
    graph.add_node("respond", respond_node)

    graph.add_edge(START, "classify")
    graph.add_edge("classify", "retrieve")
    graph.add_edge("retrieve", "memory")
    graph.add_edge("memory", "decide")
    graph.add_conditional_edges("decide", _route_after_decide, {"validate": "validate", "respond": "respond"})
    graph.add_edge("validate", "act")
    graph.add_conditional_edges("act", _route_after_act, {"decide": "decide", "respond": "respond"})
    graph.add_edge("respond", END)

    return graph.compile()


class SupportHarness:
    def __init__(self, customer_id: str, provider: GroqProvider | None = None):
        accounts = json.loads((DATA_DIR / "accounts.json").read_text(encoding="utf-8"))
        if customer_id not in accounts:
            raise ValueError(f"Unknown customer_id {customer_id!r}; cannot open a session for it.")

        self.customer_id = customer_id
        self.provider = provider or GroqProvider()
        self.retriever = _shared_retriever()
        self.long_term = LongTermMemory(DATA_DIR / "ticket_history.json")
        self.short_term = ShortTermMemory()
        self.audit_log: list[dict] = []
        self._orders_index = {
            oid: order["customer_id"]
            for oid, order in json.loads((DATA_DIR / "orders.json").read_text(encoding="utf-8")).items()
        }
        self.mcp_client: MCPToolClient | None = None  # set by __aenter__

    async def __aenter__(self) -> "SupportHarness":
        self.mcp_client = await MCPToolClient(self.customer_id).__aenter__()
        return self

    async def __aexit__(self, *exc_info):
        if self.mcp_client is not None:
            await self.mcp_client.__aexit__(*exc_info)

    def _validate_and_check_permission(self, tool_name: str, raw_args: dict) -> tuple[bool, str, dict | None]:
        """The harness-enforced boundary, called from the `validate` node.

        Returns (allowed, category, parsed_args). category is one of
        "allowed", "unknown_tool", "malformed", "out_of_scope" -- kept
        distinct so the audit log and the demo can show *why* something was
        rejected, not just that it was.
        """
        model_cls = _ARG_MODELS.get(tool_name)
        if model_cls is None:
            return False, "unknown_tool", None

        try:
            parsed = model_cls.model_validate(raw_args)
        except ValidationError:
            return False, "malformed", None
        if not parsed.is_well_formed:
            return False, "malformed", None

        if tool_name == "lookup_order":
            owner = self._orders_index.get(parsed.order_id)
            if owner is None or owner != self.customer_id:
                return False, "out_of_scope", parsed.model_dump()
            return True, "allowed", parsed.model_dump()

        if tool_name == "check_account_status":
            if parsed.customer_id != self.customer_id:
                return False, "out_of_scope", parsed.model_dump()
            return True, "allowed", parsed.model_dump()

        return False, "unknown_tool", None

    def _reason_text(self, tool_name: str, raw_args: dict, category: str) -> str:
        if category == "unknown_tool":
            return f"{tool_name!r} is not a recognized tool."
        if category == "malformed":
            return f"Arguments for {tool_name} failed schema/format validation: {raw_args!r}."
        if category == "out_of_scope":
            return (
                f"{tool_name}({raw_args!r}) does not belong to the authenticated "
                f"customer {self.customer_id}; refusing to dispatch."
            )
        return "rejected"

    async def handle_turn(self, user_text: str) -> str:
        initial_state: TurnState = {
            "messages": self.short_term.as_list() + [{"role": "user", "content": user_text}],
            "customer_id": self.customer_id,
            "ticket_type": "",
            "policy_context": "",
            "history_summary": "",
            "pending_tool_calls": [],
            "validated_calls": [],
            "iteration_count": 0,
            "final_text": None,
        }
        final_state = await _compiled_graph().ainvoke(
            initial_state,
            config={"configurable": {"harness": self}, "recursion_limit": 100},
        )
        # Persist the whole turn (user message, any tool round trips, final
        # answer) back into the conversation buffer for the next turn.
        self.short_term.messages = final_state["messages"]
        return final_state["final_text"] or ""
