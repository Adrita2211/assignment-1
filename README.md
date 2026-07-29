# E-commerce Order Support Agent

## 1. What this is

A multi-turn support agent for a fixed e-commerce domain: order status,
delivery issues, refund requests, and subscription/account questions. You
chat with it as an authenticated customer; on each message it classifies the
ticket, retrieves relevant policy text (hybrid lexical + semantic search over
a small policy doc set), decides whether it needs real order/account data
(and if so, proposes a tool call that a harness validates and permission-checks
*before* it's allowed to run), and replies grounded in whatever it actually
retrieved or looked up -- not from memorized/invented policy numbers. It
remembers the current conversation (short-term) and this customer's prior
support tickets (long-term), across turns. It does **not** issue refunds --
that's the one irreversible action in this domain, intentionally out of
scope.

## 2. Prerequisites

- **Python 3.11 or newer**, installed and on your `PATH`.
- **A free Groq API key** -- sign up at https://console.groq.com/keys (no
  paid tier needed). The agent uses Groq to run the LLM (`openai/gpt-oss-120b`).
- **Internet access on first run only**, to download a small (~90MB) local
  embedding model (`all-MiniLM-L6-v2`) used for semantic search. It's cached
  locally after that -- no internet needed on later runs. If the download is
  ever unavailable, the agent still works; it just falls back to lexical-only
  search and prints a warning.
- **No database, no Docker, no external services required.** The vector
  index (FAISS) and MCP server both run as local, in-process/subprocess
  components -- nothing to install or start separately.
- **Windows only:** if `pip install` fails partway through with an error
  mentioning a very long file path (`OSError: ... No such file or
  directory ...`), that's Windows' path-length limit, not a broken package --
  clone the repo into a short path (e.g. `C:\dev\...`) rather than somewhere
  deeply nested (e.g. deep inside `Downloads` or a synced cloud folder), or
  enable Windows Long Path support.

## 3. Setup

Run these in order, from a terminal:

```bash
git clone <this-repo-url>
cd ecommerce-support-agent
python -m venv .venv
```

Activate the virtual environment (pick the line for your shell):

```bash
.venv\Scripts\Activate.ps1      # Windows, PowerShell
.venv\Scripts\activate.bat      # Windows, cmd.exe
source .venv/bin/activate       # macOS / Linux
```

Then:

```bash
pip install -r requirements.txt
```

```bash
cp .env.example .env            # macOS / Linux
copy .env.example .env          # Windows
```

Open `.env` in a text editor and set the one variable it defines:

```
GROQ_API_KEY=your_actual_key_here
```

That's the exact variable name the code reads (`agent/provider.py`). If you
forget this step, both entry points below fail immediately with a clear
message telling you to do it -- not a stack trace.

## 4. How to run it

Two ways to run the agent. **Start with the scripted demo** -- it needs no
typing and shows every major behavior in one pass, with full internal
logging printed to the console (what got classified, what was retrieved,
what the harness allowed or rejected):

```bash
python demo.py
```

Once that runs cleanly, try the interactive mode -- a real back-and-forth
conversation:

```bash
python main.py
```

It prints a list of mock customer IDs (`CUST001` through `CUST005`) and asks
you to log in as one. Type messages at the `you>` prompt; type `exit` to
quit. Example first message: `Where is my order ORD1002?` (log in as
`CUST002` for that one -- see `data/orders.json` for which order belongs to
which customer).

## 5. Project structure

```
main.py                    interactive CLI -- start a chat session as a chosen customer
demo.py                    scripted run-through of every scenario, no typing needed
requirements.txt           Python dependencies
.env.example                template for the one required environment variable
mcp_server/server.py       the MCP server -- exposes lookup_order and check_account_status
agent/harness.py           the harness loop: classify -> retrieve -> decide -> validate/permission-check -> act -> respond
agent/provider.py          Groq API wrapper (the only file that imports the groq SDK)
agent/mcp_client.py        spawns mcp_server/server.py and talks to it over MCP (stdio)
agent/rag.py               hybrid retriever: BM25 (lexical) + FAISS/sentence-transformers (semantic) over policies/
agent/classify.py          rule-based ticket-type classifier
agent/memory.py            ShortTermMemory (this conversation) and LongTermMemory (this customer's ticket history)
data/orders.json           mock order database
data/accounts.json         mock account database (active / flagged / suspended)
data/ticket_history.json   mock prior-ticket history, keyed by customer_id
policies/*.md              the policy documents the RAG layer retrieves from
```

If you only read three files to understand how this works, read
`agent/harness.py` (the loop and the permission boundary),
`mcp_server/server.py` (the tools it can call), and `agent/rag.py` (how it
retrieves policy grounding).

## 6. Why I built the harness this way

**The harness, not the model, decides.** `SupportHarness.handle_turn` in
`agent/harness.py` sends every model-proposed tool call through
`_validate_and_check_permission()` before any of them reach
`agent/mcp_client.py`. A rejected call never touches the MCP layer at all --
it gets a synthetic `tool_result` explaining why (with a machine-readable
`category`), and the conversation continues. That's the one line to point to
for grading: `agent/harness.py`, inside `handle_turn`, the `for tc in
result["tool_calls"]:` loop, calling `self._validate_and_check_permission(...)`
before `self.mcp_client.call_tool(...)`.

**Validation and security, layered, in the order a call actually passes
through** (see `_validate_and_check_permission`):
1. **Tool allowlist** -- the name must be one of the two registered tools;
   anything else is `unknown_tool`, rejected immediately.
2. **Schema/shape validation** -- a pydantic model per tool
   (`LookupOrderArgs`, `CheckAccountStatusArgs`) with `extra="forbid"`
   rejects missing fields, wrong types, *and* unexpected extra keys (e.g. a
   model trying to smuggle an `admin_override` field through). This runs
   independently of, and before, whatever the MCP server's own schema would
   also reject -- two separate layers, not one shared check.
3. **ID-format validation** -- `order_id`/`customer_id` must match
   `ORD\d+`/`CUST\d+` via regex, catching garbage input cheaply before it's
   even worth resolving ownership.
4. **Ownership check** -- the resolved order/customer must belong to *this*
   ticket's authenticated `customer_id`, checked against `data/orders.json`
   loaded directly by the harness (not fetched through the tool).

Every decision is appended to `self.audit_log` (tool, args, allowed/rejected,
category, reason), and a `MAX_TOOL_ITERATIONS` cap (6) stops a
misbehaving/adversarial model from looping tool calls indefinitely in one
turn -- a bounded-resources safety valve, not something a normal
conversation should ever hit.

**1. Scoping ticket types.** I used the four categories the assignment names
(order status, delivery issue, refund request, subscription/account) as
fixed, mutually-exclusive-ish buckets and classify with a deterministic
keyword heuristic (`agent/classify.py`) rather than an LLM call. The
categories are lexically distinct in how customers actually phrase them
("where is my order" vs. "refund" vs. "suspended"), and a deterministic
classifier is auditable -- the same input always produces the same category,
with no extra API round trip or risk of the classifier itself hallucinating
a category. Anything that doesn't match falls into `general` rather than
being forced into one of the four.

**2. A specific permission-boundary decision in the MCP layer.** `order_id`
does not map 1:1 to the current customer by construction -- any customer
could type any order ID in chat. So the harness resolves `order_id` to its
owning `customer_id` using its own copy of the order data and compares that
owner to the ticket's authenticated `customer_id`; only a match reaches
`agent/mcp_client.py`. Each MCP server subprocess is additionally bound to a
`SESSION_CUSTOMER_ID` environment variable at spawn time (one subprocess per
ticket, in `MCPToolClient.__aenter__`), and raises
`mcp.server.fastmcp.exceptions.ToolError` (proper `isError=True` semantics)
for both malformed IDs and cross-customer access -- defense in depth, not
the primary boundary. I deliberately did not implement the check as an `if`
inside the tool function alone; the harness enforcing it before dispatch is
what makes it a real boundary rather than something bolted on afterward.

**RAG grounding is logged, not just claimed, and hybrid to reduce
hallucination.** `agent/rag.py`'s `HybridPolicyRetriever` fuses two signals
per query:
- **BM25** (lexical, standard-library only) -- a strictly better lexical
  ranker than plain TF-IDF cosine, but still capped at scoring shared
  vocabulary.
- **Dense vector similarity** (semantic) -- a local `sentence-transformers`
  model (`all-MiniLM-L6-v2`) embeds every policy doc, indexed in FAISS
  (`IndexFlatIP`, exact search), so a question that rephrases a policy with
  *zero* shared words can still be found.

They're fused 20% lexical / 80% vector by default (shifted to 100% BM25 if
the embedding backend can't load) -- calibrated against a held-out set of
genuine vs. adversarial-but-plausible off-topic questions, because a 50/50
split let off-topic queries through whenever they happened to share one real
word with a doc. Anti-hallucination is enforced *at retrieval time*, not
left to the model: `retrieve()` only returns docs whose fused score clears
`DEFAULT_MIN_FUSED_SCORE` -- below that, it returns nothing, and the system
prompt instructs the model to say the question isn't covered rather than
guess. Every turn prints `[RAG] retrieved: [...]` or `[RAG] no relevant
chunk found -- honest gap` to the console, so which chunk(s) justified an
answer is checkable, not just claimed.
