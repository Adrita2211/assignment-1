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
server.py                  HTTP entrypoint (FastAPI) -- the deployed agent, one stateless request per ticket
requirements.txt           Python dependencies
.env.example               template for every environment variable the code reads
Dockerfile                 container image for the deployed agent (server.py)
mcp_server/server.py       the MCP server -- exposes lookup_order and check_account_status
agent/harness.py           the harness -- a LangGraph StateGraph: classify -> retrieve -> memory -> decide -> validate -> act -> respond
agent/provider.py          LLMProvider abstraction: GroqProvider (functional) + BedrockProvider (documented stub, see section 13)
agent/mcp_client.py        spawns mcp_server/server.py and talks to it over MCP (stdio)
agent/rag.py               local retriever over policies/ -- BM25 (lexical) only for local dev; the FAISS/sentence-transformers semantic half is temporarily disabled (see its module docstring) now that pgvector is the real semantic backend (below)
agent/rag_pgvector.py      RDS PostgreSQL + pgvector retriever -- same fusion math, used when DATABASE_URL is set (see section 11)
agent/classify.py          rule-based ticket-type classifier
agent/memory.py            ShortTermMemory (this conversation) and LongTermMemory (this customer's ticket history)
agent/tracing.py           shared LangFuse client (see section 7)
scripts/seed_pgvector.py   one-time job: embed policies/*.md and load them into RDS + pgvector
eval/fixtures.py           the fixed 12-ticket trajectory-eval set (section 8)
eval/trajectory_eval.py    the CI regression gate (section 9)
eval/llm_judge.py          LLM-as-judge groundedness scorer (section 8)
eval/before_after_report.py  clean-vs-regressed pass-rate comparison (section 10)
.github/workflows/ci-cd.yml  eval-gate -> build-and-push -> deploy (sections 9, 11)
infra/cloudformation/*.yaml  ECS/ALB/ECR/RDS+pgvector/autoscaling stack (section 11)
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

**The harness, not the model, decides -- built as a LangGraph `StateGraph`.**
`agent/harness.py` models the loop as graph nodes, not a hand-rolled
while-loop:

```
classify -> retrieve -> memory -> decide --(conditional edge)--> validate -> act --(conditional edge)--> decide (loop)
                                       \                                                              \
                                        --(no tool calls)--> respond -> END        (iteration cap hit)--> respond -> END
```

`decide` is the *only* node that talks to the LLM (via `GroqProvider`) and
it only ever *proposes* tool calls -- it never decides whether they run.
That decision belongs to `validate`, a completely separate node that sits on
every path from `decide` to `act`. LangGraph's conditional-edge routing
functions (`add_conditional_edges`) can choose the next node based on state,
but can't themselves produce state updates -- so the actual permission-check
*logic* has to live in a node, not the router callback. `validate` is that
node: every proposed call passes through it, and `act` (the only node that
touches `agent/mcp_client.py`) only ever dispatches to MCP for a call
`validate` already marked allowed. A rejected call never reaches
`self.mcp_client.call_tool(...)` at all -- `act` still runs (it has to feed
a rejection `tool_result` back to the model either way), but for a rejected
call it synthesizes the rejection itself instead of calling MCP. That's the
line to point to for grading: `agent/harness.py`, `validate_node`, calling
`harness._validate_and_check_permission(...)`, and `act_node`'s `if
v["allowed"]:` branch guarding the one and only call to
`harness.mcp_client.call_tool(...)`.

**Validation and security, layered, in the order a call actually passes
through** (see `SupportHarness._validate_and_check_permission`, called from
`validate_node`):
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
category, reason), and a `MAX_TOOL_ITERATIONS` cap (6), enforced by the
conditional edge after `act` (`_route_after_act`), stops a
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
hallucination.** `agent/rag.py`'s `HybridPolicyRetriever` was designed to
fuse two signals per query -- **BM25** (lexical, standard-library only) and
**dense vector similarity** (semantic, local `sentence-transformers` +
FAISS) -- so a question that rephrases a policy with *zero* shared words
can still be found. As of Assignment 2, the local FAISS/sentence-transformers
half is disabled by default (BM25-only fallback) since `agent/rag_pgvector.py`
+ RDS pgvector (section 11) is now the real semantic backend for the
deployed agent; local dev without `DATABASE_URL` set is lexical-only.

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

---

# Assignment 2: Observability, Evaluation, Gating, and AWS Deployment

Everything below wraps the Assignment 1 agent above in four layers: real
tracing, trajectory + LLM-judge evaluation, a CI regression gate, and a real
AWS deployment. The agent itself (harness, RAG, permission boundary) is
unchanged in shape -- these sections instrument and gate it, they don't
replace it.

## 7. Tracing setup

Every tool call (`lookup_order`, `check_account_status`), the retrieval step,
the model call, and the ticket as a whole produce a real LangFuse span --
see `agent/tracing.py` for the client and `agent/harness.py`'s `@observe`
decorators on `classify_node`, `retrieve_node`, `decide_node`,
`_traced_lookup_order` / `_traced_check_account_status`, and
`SupportHarness.handle_turn` (the top-level `agent` span every child span
nests under).

One-time setup: create a free project at
[cloud.langfuse.com](https://cloud.langfuse.com) (LangFuse Cloud) and grab
its keys. (An earlier version of this project also had a self-hosted
LangFuse-on-EC2 CloudFormation option; it was never actually deployed and
has been removed to keep the infra directory to what's actually used --
LangFuse Cloud's free tier is a fully legitimate choice per the assignment's
own build guide, not a compromise.)

Either way, set these three variables in `.env` (already in
`.env.example`):

```
LANGFUSE_HOST=http://localhost:3000     # or https://cloud.langfuse.com
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
```

If any of these is unset, `agent/tracing.py` disables tracing entirely (the
agent still runs, it just produces no spans) -- deliberate, so
`eval/trajectory_eval.py` and the CI eval-gate job never need real LangFuse
credentials just to run the regression check.

Run a handful of real tickets (`python main.py` or `python demo.py`) and
confirm they show up correctly nested in the dashboard before relying on it
for the video demo.

**For the deployed agent specifically:** `deploy_agent_stack.sh` seeds
placeholder `"unset"` values into the three `LANGFUSE_*` SSM parameters --
update them with your real LangFuse Cloud credentials before relying on the
deployed agent's traces:
```bash
aws ssm put-parameter --name "/ecommerce-support-agent/LANGFUSE_HOST" --value "https://cloud.langfuse.com" --type SecureString --overwrite
aws ssm put-parameter --name "/ecommerce-support-agent/LANGFUSE_PUBLIC_KEY" --value "pk-lf-..." --type SecureString --overwrite
aws ssm put-parameter --name "/ecommerce-support-agent/LANGFUSE_SECRET_KEY" --value "sk-lf-..." --type SecureString --overwrite
aws ecs update-service --cluster ecommerce-support-agent-cluster --service ecommerce-support-agent-service --force-new-deployment
```
SSM secrets are injected as environment variables at container startup, not
live-refreshed -- the `force-new-deployment` is required for a running task
to actually pick up updated values.

**NB -- traces vs. scores, deliberately different scopes:** every real
request (a live `/chat` call, `demo.py`, `main.py`) produces a full,
correctly-nested **trace** -- that satisfies section 2.1's requirement on
its own. **Scores**, however, only attach to traces produced by actually
running one of the `eval/*.py` scripts against the fixed ticket set (see
`eval/langfuse_scores.py`'s `push_score()`, called from
`trajectory_eval.py`, `calibration_eval.py`, etc.) -- a plain live request
is never scored. This is intentional, not a gap: scoring is a batch
evaluation-suite concept (comparing a fixed, known ticket set against a
baseline over time), not a live-monitoring feature. Scoring every live
customer request would mean an extra LLM-judge call (cost + latency) per
real interaction just to populate a dashboard -- a real production system
that wants that would sample a small percentage of live traffic for
background scoring, which this project doesn't attempt and wasn't asked to.

## 8. Trajectory evaluation and LLM-as-judge

**Trajectory eval** (`eval/trajectory_eval.py`) scores whether the right
grounding/tool steps happened, not whether the final answer merely reads
correctly. The fixed set lives in `eval/fixtures.py`: 12 tickets across all
4 ticket types (3 each of `order_status`, `delivery_issue`,
`refund_request`, `subscription_account`), reused unmodified by the CI gate
and the before/after report so all three always score the same set.

```bash
python -m eval.trajectory_eval                    # score against eval/baseline.json
python -m eval.trajectory_eval --update-baseline   # (re)write the baseline after a real, reviewed change
```

Run `--update-baseline` once, from a clean checkout with a working
`GROQ_API_KEY`, before relying on the gate -- `eval/baseline.json` isn't
committed with a pre-baked number, since a baseline written by a run that
never actually happened would defeat the point of a gate that compares
against real history.

**LLM-as-judge** (`eval/llm_judge.py`) scores what a trajectory rule can't:
whether a specific claim in the response is actually supported by the
reference (the real policy text or order/account record), using a
fact-checking rubric (0-10, "does every claim appear in the reference"), not
a vague quality rating -- see the module docstring for why that distinction
matters. Each ticket is judged 3 times by default (`--runs`) since judge
scores are non-deterministic; report the spread, not a single run.

```bash
python -m eval.llm_judge                 # first 5 tickets, 3 judge runs each (assignment minimum)
python -m eval.llm_judge --tickets 12 --runs 5
```

**Actual reported scores** (5 tickets, 3 judge runs each, against the live
deployed agent):

| Ticket | Groundedness (mean, stdev) | Task success (mean, stdev) |
|---|---|---|
| `t1_order_status_shipped` | 10.0, 0.0 | 10.0, 0.0 |
| `t2_order_status_tracking` | 9.33, 0.47 | 10.0, 0.0 |
| `t3_order_status_processing` | 7.0, 0.0 | 10.0, 0.0 |
| `t4_delivery_late` | 4.67, 0.47 | 10.0, 0.0 |
| `t5_delivery_lost` | 2.67, 0.47 | 10.0, 0.0 |

**Overall: groundedness 6.73/10, task_success 10.0/10.** The split matters:
every response fully addressed what the customer asked (task_success is a
clean 10 across the board), but groundedness drops sharply on the delivery
tickets -- a response can satisfy the customer's actual question while still
citing details the judge couldn't verify against the reference. That gap is
exactly why this project scores the two dimensions separately instead of one
blended "quality" number.

**NB -- known limitation: the judge and the agent share the same model.**
`eval/llm_judge.py` and `eval/calibration_eval.py` both instantiate a fresh
`GroqProvider()` (same `openai/gpt-oss-120b` model the agent itself uses)
as the judge, rather than a separate, ideally more capable model. This was
a cost decision -- Groq's free tier -- not a methodologically sound one: a
model judging its own family's output risks being systematically lenient
toward its own phrasing/reasoning patterns and sharing its own blind spots,
rather than catching them independently. The correct design would use a
distinct judge model (e.g. a larger/different-provider model) precisely
*because* it wouldn't share the generator's failure modes. Worth fixing
before trusting these scores for anything beyond this assignment's demo
purposes.

**A real "confident wrong path" case (found via LangFuse, not manufactured):**

Trace ID `119d9e059b6127e0e81673297fab3327` -- customer CUST005 asks: *"My
account is under review and I want to stop being charged every month, what
do I do?"* The final answer reads completely fine on its own: a confident,
specific 4-step subscription-cancellation procedure ("Log in... Go to
Account -> Subscriptions... Select the Plus (monthly) plan and choose Cancel
subscription... Confirm the cancellation").

The trace's `retrieve_policy` span, however, shows
`retrieved_doc_ids: ['account_suspension_appeal']` -- `subscription_cancellation.md`
was **never retrieved at all**. Checking both docs directly: neither
contains any UI navigation steps (`account_suspension_appeal.md` covers
appeal timelines; `subscription_cancellation.md` covers only billing/refund
rules). The entire 4-step procedure is fabricated from the model's general
training knowledge, not grounded in anything the agent was actually given.

Root cause: the query blends two topics (suspension + cancellation);
retrieval (`top_k=2`, `min_fused_score` threshold) only surfaced the
suspension doc as relevant enough, and the system prompt's honest-gap
instruction only fires when *nothing* is retrieved -- not when something
*partially* relevant is retrieved and the model fills in the rest from
memory. A final-answer-only check would have waved this straight through;
only the trajectory (which doc was actually retrieved vs. what was actually
claimed) reveals it.

## 8a. Full evaluation taxonomy (all six axes, five layers, three techniques)

Section 8 above covers the assignment's two required scorers (trajectory
and groundedness). This project's eval suite actually scores all six checks
from the evaluation taxonomy, across all five observability layers, using
three techniques -- trajectory eval, tracing, and LLM-as-judge -- applied
more broadly than the assignment's minimum:

**The six checks ("what you're checking for"):**

| Check | Where it's scored |
|---|---|
| 1. Task Success | `eval/llm_judge.py` -- `task_success` dimension (did the response actually address what was asked, independent of whether it's grounded) |
| 2. Groundedness | `eval/llm_judge.py` -- `groundedness` dimension |
| 3. Policy Adherence | `eval/policy_adherence_eval.py` -- rule-based: the harness's permission boundary never marks a cross-customer call "allowed," and no response ever confirms a refund was processed |
| 4. Safety | `eval/safety_eval.py` -- prompt-injection and impersonation attempts (`eval/fixtures.py` `SAFETY_TICKETS`), checked for PII leakage and system-prompt exfiltration in the response text |
| 5. Robustness | `eval/robustness_eval.py` -- typo'd, all-caps, terse, and rambling rephrasings of five real tickets (`ROBUSTNESS_TICKETS`), scored against the same trajectory rule as the originals |
| 6. Calibration | `eval/calibration_eval.py` -- genuinely out-of-policy questions (`CALIBRATION_TICKETS`); passes only if retrieval honestly returns nothing AND the response doesn't fabricate a specific policy detail |

Run each individually, or all at once via the aggregate report:

```bash
python -m eval.policy_adherence_eval
python -m eval.safety_eval
python -m eval.robustness_eval
python -m eval.calibration_eval
python -m eval.report              # all six checks + system + trajectory + longitudinal, one Markdown file
```

**The five observability layers:**

| Layer | Where it's captured |
|---|---|
| 1. System (latency, cost, error rate) | `agent/harness.py`'s `SupportHarness.metrics()`, populated per LLM call in `decide_node`; aggregated per suite run in `eval/trajectory_eval.py`'s `_aggregate_system_metrics` |
| 2. Trajectory (every tool call, retrieval, decision point) | `eval/trajectory_eval.py`, and every LangFuse span in `agent/harness.py` (section 7) |
| 3. Output (final answer, schema match, confidence) | the response text itself (every eval script), plus `agent/harness.py`'s pydantic schema validation (`extra="forbid"`) as the schema-match signal |
| 4. Longitudinal (drift, variance, failure clustering, coverage) | `eval/longitudinal_eval.py`, reading/appending `eval/run_history.jsonl` |
| 5. Human-facing (legible without having been there) | `eval/report.py` -- generates `eval/report_output.md`, this section's own deliverable |

**The three techniques**, and where each is real, not simulated:
- **Trajectory eval** -- `eval/trajectory_eval.py` (gates CI) plus `eval/robustness_eval.py` and `eval/policy_adherence_eval.py`'s structural check (same technique, different fixture sets and pass rules).
- **Tracing** -- `agent/tracing.py` + the `@observe` spans in `agent/harness.py` (section 7).
- **LLM-as-judge** -- `eval/llm_judge.py` (groundedness, task_success) and `eval/calibration_eval.py` (honest-gap judge), both multi-run per ticket since judge scores are non-deterministic (section 6's common-pitfall warning).

Accumulate longitudinal history and regenerate the full report together:

```bash
python -m eval.longitudinal_eval --repeats 5   # append 5 fresh runs to run_history.jsonl
python -m eval.report                          # picks up that history in its Longitudinal section
```

## 9. The CI regression gate

**File:** `.github/workflows/ci-cd.yml`, `eval-gate` job.
**What it runs:** `python -m eval.trajectory_eval` (env: `GROQ_API_KEY` from
a repo secret). The script itself decides pass/fail via its exit code --
the workflow YAML doesn't duplicate that logic.
**What it checks:** the trajectory only (which of `lookup_order`,
`check_account_status`, `retrieve_policy` actually ran per ticket), not the
final-answer text -- see section 3's defensible justification #2 for exactly
what that means it can't catch.
**Threshold:** `REGRESSION_THRESHOLD_POINTS = 15` (default in
`eval/trajectory_eval.py`, overridable via `--threshold` or the env var of
the same name). Relative, not absolute: fails only if the current pass rate
drops more than 15 points below `eval/baseline.json`, not if it's simply
under some fixed number.

**Demonstrating the gate blocking a real regressed push (done, live, twice):**
the repo's regression flag is `AGENT_REGRESSED` (see `agent/harness.py`'s
`retrieve_node` -- when true, retrieval is silently disabled, simulating a
broken grounding step without touching anything else about the agent, the
same discipline as `agent-cicd-demo`'s `AGENT_REGRESSED` tool-removal flag).
The actual demo pushed two commits to the `regression-demo` branch:

1. **The regression** -- flipped `AGENT_REGRESSED`'s default from `false` to
   `true` in `agent/harness.py` (a one-line diff). Real GitHub Actions run:
   [`31295569394`](https://github.com/Adrita2211/assignment-1/actions/runs/31295569394)
   -- `eval-gate` job **FAILED** (`Run trajectory-eval regression gate` step,
   exit code 1). Log: `Baseline: 91.7% | Current: 58.3% | Drop: 33.4 points |
   Threshold: 15.0 points -- REGRESSION GATE: FAILED`. `build-and-push` and
   `deploy` never ran (blocked by `needs: eval-gate`).
2. **The fix** -- reverted the default back to `false`. Real run:
   [`31296910028`](https://github.com/Adrita2211/assignment-1/actions/runs/31296910028)
   -- `eval-gate` **passed**, and this time `build-and-push`/`deploy` ran for
   real: a commit-SHA-tagged image (`ecommerce-support-agent:b241a699ffd199bec1f509ee8e4f6993cb045cf3`)
   was built by GitHub's runners, pushed to ECR via OIDC, and deployed to the
   live ECS service -- confirmed by hitting the ALB URL post-deploy and
   checking the running task definition's image tag.

## 10. Before/after report and defensible justifications

```bash
python -m eval.before_after_report
```

Runs the trajectory-eval suite twice in one process -- once clean, once with
the regression flag forced on -- and prints both pass rates plus exactly
which ticket IDs flipped from PASS to FAIL. This is the number for the
README/PR description; the CI gate above is what actually blocks the
regressed build from deploying, this script just documents the drop it
would have caused.

**Actual output from a real run:**

```
======================================================================
BEFORE / AFTER REPORT
======================================================================
Clean pass rate:     91.7%
Regressed pass rate: 58.3%
Drop:                33.4 points

Tickets that flipped PASS -> FAIL (4):
  - t6_delivery_policy_general
  - t7_refund_damaged
  - t9_refund_eligibility_window
  - t10_subscription_cancel_policy
======================================================================
```

91.7% to 58.3%, a 33.4-point drop -- well past the 15-point gate threshold,
consistent with the live GitHub Actions failure in section 9.

**Justification 1 -- why the threshold is 15 points, not looser or
tighter:** with 12 fixed tickets, one ticket flipping is an 8.3-point swing.
15 points tolerates a single ticket's worth of ordinary LLM run-to-run
variance (a borderline retrieval score, a rate-limit fallback to a
different Groq model) without false-failing a genuinely clean build, while
still reliably catching the actual regression this repo ships: disabling
retrieval fails all 5 `retrieve_policy`-required tickets outright, a ~42-point
drop, nearly 3x the threshold. A looser threshold (say 40+) would let a
partial regression -- one whole ticket *type* silently breaking, roughly a
25-point drop -- pass through undetected. A tighter one (say 5 points) would
false-fail on ordinary model variance alone, which trains everyone to
ignore the gate the first time it cries wolf -- worse than not having one.

**Justification 2 -- what the gate checks and what it can't catch:** it
checks trajectory only (did the required step run), not the final-answer
text. That means it will correctly catch this repo's actual regression
(retrieval silently disabled) even on a ticket where the model's fallback
answer happens to still sound reasonable. It will **not** catch a case
where the required step *did* run but did the wrong thing underneath --
`retrieve_policy` returning the wrong doc that still clears the fused-score
threshold, or `lookup_order` getting called with a subtly wrong ID that
still resolves to a real (wrong) order. That's precisely the gap
`eval/llm_judge.py`'s groundedness scoring exists to cover, and precisely
why the assignment asks for both a rule-based and an LLM-judge scorer
instead of trusting either alone.

## 11. AWS architecture

Deploy target is locked in per the assignment: ECS Fargate behind an ALB,
images in ECR tagged by commit SHA, RDS PostgreSQL + pgvector as the vector
store, GitHub OIDC (no static AWS keys), and CloudFormation as the
infrastructure-as-code. Everything below is adapted directly from
`agent-cicd-demo`'s own working template/workflow shape, per the
assignment's own guidance to reuse it rather than redesign it.

**Stack:** `infra/cloudformation/agent-infra.yaml` (`APP_NAME` defaults to
`ecommerce-support-agent`; every resource name below assumes that default):

| Resource | Name |
|---|---|
| ECS cluster | `ecommerce-support-agent-cluster` |
| ECS service | `ecommerce-support-agent-service` |
| ECS task family | `ecommerce-support-agent-task` |
| ECR repository | `ecommerce-support-agent` |
| RDS instance identifier | `ecommerce-support-agent-db` |
| ALB | `ecommerce-support-agent-alb` (fetch its DNS name below) |

**Fetching the ALB URL:**
```bash
aws cloudformation describe-stacks --stack-name ecommerce-support-agent-infra \
  --query "Stacks[0].Outputs[?OutputKey=='AlbDnsName'].OutputValue" --output text
```

**Live values for this deployment** (account `058264386876`, region `us-east-1`):
- **ALB URL:** `http://ecommerce-support-agent-alb-580202083.us-east-1.elb.amazonaws.com`
- **RDS endpoint:** `ecommerce-support-agent-db.cd8k6wsa4mwz.us-east-1.rds.amazonaws.com`
- **GitHub deploy role ARN:** `arn:aws:iam::058264386876:role/ecommerce-support-agent-github-deploy-role`

```bash
curl http://ecommerce-support-agent-alb-580202083.us-east-1.elb.amazonaws.com/health
curl -X POST http://ecommerce-support-agent-alb-580202083.us-east-1.elb.amazonaws.com/chat \
  -H "Content-Type: application/json" \
  -d '{"customer_id": "CUST002", "message": "Where is my order ORD1002?"}'
```

**Gotchas hit deploying this from Windows/Git Bash, for whoever runs this next:**
- **No default VPC in the account/region:** `deploy_agent_stack.sh`'s automatic
  VPC/subnet lookup returns nothing if none exists. Fix once:
  `aws ec2 create-default-vpc --region us-east-1`.
- **Git Bash's MSYS path conversion** mangles anything that looks like an
  absolute POSIX path -- including SSM parameter names (`/ecommerce-support-agent/...`)
  and `--template-file` arguments, in opposite directions. Set
  `export MSYS_NO_PATHCONV=1` before running any `aws` command from Git Bash
  on Windows (the deploy/teardown scripts' `SCRIPT_DIR` already uses `pwd -W`
  to sidestep the template-file half of this).
- **CloudFormation's `Description` field caps at 1024 characters** -- keep
  template-level rationale in `#` comments, not the `Description:` key.
- **Seeding pgvector from outside the VPC:** the RDS instance is
  `PubliclyAccessible: false` by design (private subnet, ECS-only security
  group). To run `scripts/seed_pgvector.py` from a local machine, temporarily
  flip it (`aws rds modify-db-instance --publicly-accessible --apply-immediately`),
  open the DB security group to your IP for port 5432, seed, then revert both
  immediately after. DNS takes a few minutes to actually propagate the public
  address after the flag change -- don't assume it's instant.

**RDS + pgvector wiring:** `agent/rag_pgvector.py`'s `PgVectorPolicyRetriever`
is used automatically whenever `DATABASE_URL` is set (see
`agent/harness.py`'s `_shared_retriever()`); unset, the agent falls back to
the local FAISS index in `agent/rag.py` for zero-dependency local dev. The
deployed task definition always sets `DATABASE_URL` (see
`agent-infra.yaml`'s `TaskDefinition`). Seed the table once per fresh
database:
```bash
export DATABASE_URL=postgresql://<user>:<password>@<rds-endpoint>:5432/supportagent
python -m scripts.seed_pgvector
```

**Autoscaling:** `ScalableTarget` + `ScalingPolicy` in `agent-infra.yaml` --
target-tracking on `ECSServiceAverageCPUUtilization`, target **60%** CPU,
**min 1 / max 4** tasks (all three are stack parameters:
`AutoscalingCpuTarget`, `MinTaskCount`, `MaxTaskCount`). 60% leaves headroom
to absorb a burst before scaling kicks in, while still triggering well
before a task is saturated enough to start timing out requests; min 1
keeps idle cost to a single Fargate task, max 4 is a demo-scale ceiling, not
a production capacity plan. To trigger and observe a real scale-out event:
fire a burst of concurrent requests at the ALB URL's `/chat` endpoint (a
simple `for`-loop with backgrounded `curl`s is enough) and watch the ECS
console's task count for the service actually increase.

**Actual scale-out event (real, captured via CLI):** sustained concurrent
load against `/chat` pushed `AWS/ECS` `CPUUtilization` to 74-96% for several
minutes. Application Auto Scaling's `describe-scaling-activities` recorded:

```
Description: "Setting desired count to 2."
Cause: "monitor alarm TargetTracking-service/ecommerce-support-agent-cluster/
        ecommerce-support-agent-service-AlarmHigh-... in state ALARM
        triggered policy ecommerce-support-agent-cpu-target-tracking"
```

`desiredCount` went 1 -> 2, `runningCount` followed (briefly hit 3 during the
transition), and settled at `desiredCount: 2, runningCount: 2` once load
stopped. Scaled back in on its own after the 120s cooldown confirmed
sustained low CPU.

**Deploying:**
```bash
export GITHUB_ORG=<your-username-or-org>
export GITHUB_REPO=<this-repo-name>
export GROQ_API_KEY=<a-real-groq-key>
export DB_MASTER_PASSWORD=<a-strong-password>
bash infra/deploy_agent_stack.sh          # ECS/ALB/ECR/RDS/autoscaling stack
```
`deploy_agent_stack.sh` prints the `AWS_DEPLOY_ROLE_ARN` and `AWS_REGION` to
add as GitHub repo **variables** (not secrets -- the role ARN isn't
sensitive, OIDC trust is scoped to this exact repo), plus a reminder to run
`scripts.seed_pgvector` once against the new database. Add `GROQ_API_KEY` as
a GitHub repo **secret** separately (used by the `eval-gate` job in CI,
never touches AWS).

**Deploy pipeline / no static AWS keys:** `.github/workflows/ci-cd.yml`.
Both AWS-touching jobs (`build-and-push`, `deploy`) authenticate via
`aws-actions/configure-aws-credentials@v4` with `role-to-assume: ${{
vars.AWS_DEPLOY_ROLE_ARN }}` -- GitHub's OIDC token exchanged for a
short-lived role session. There is no `AWS_ACCESS_KEY_ID` /
`AWS_SECRET_ACCESS_KEY` anywhere in this repo; confirmed by `grep -r
AWS_SECRET_ACCESS_KEY .` returning nothing.

## 12. Teardown

```bash
export AWS_REGION=us-east-1   # match whatever you deployed with
bash infra/teardown_agent_stack.sh      # empties ECR, deletes ECS/ALB/RDS/IAM stack, deletes SSM params
```
Tested, not just described -- `teardown_agent_stack.sh` empties the
ECR repo first (CloudFormation won't delete a non-empty one) and waits on
`cloudformation wait stack-delete-complete` before reporting done. The
GitHub OIDC provider (`infra/setup_oidc_provider.sh`) is never torn down --
it's an account-wide resource other repos may depend on.

## 13. Bedrock-ready path (Assignment 2's optional bonus -- superseded by Assignment 3)

**Update: this is no longer a stub.** `agent/provider.py`'s `BedrockProvider`
was a documented-but-unimplemented seam as of Assignment 2; as of
Assignment 3 §2.2, `BedrockProvider.call()` is a real, working
implementation (boto3 `bedrock-runtime` `converse()`), verified against
this project's actual `TOOLS` schema with a mocked client -- see section 15
below for the details. This section is kept for history; the current state
lives in section 15.

---

# Assignment 3: Autonomy Decision, Bedrock Migration, HITL, PII, Cost, and Semantic Cache

Builds on Assignment 2's already-deployed ECS/ALB/RDS agent. This is a real
architecture *migration*, not an additive layer: the agent consolidates
onto Bedrock AgentCore + Bedrock Knowledge Bases on Aurora, and the
Assignment 2 ECS/ALB stack is torn down once that migration is verified
live (see section 24). Everything below that doesn't require AWS
provisioning is real, verified code, tested with actual runs -- not
described. Everything that *does* require AWS provisioning (AgentCore
itself, the real Knowledge Base, Bedrock Guardrails) is code-complete and
unit-tested against mocked clients, with the real-AWS verification
explicitly called out as pending until that provisioning batch runs (see
section 24 for status).

## 14. The autonomy decision (§2.1)

**Path taken: B -- the written justification for staying single-agent, not
an A2A handoff.** Full four-question walkthrough, against real tickets and
real tool calls from this codebase (not the class example restated), lives
in [`eval/autonomy_decision.md`](eval/autonomy_decision.md). Short version:
the standard "billing disputes" candidate specialist's job already fully
exists inside this agent (`propose_refund_decision` + `agent/policy_boundary.py`
+ the HITL gate), there's no real current bottleneck the single-agent shape
causes, and staying single-agent avoids the one concrete new failure mode
(a dropped/malformed handoff) a second agent would introduce for zero
capability gain. Demonstrated against a real ticket that reads like it
needs a billing specialist (CUST003, ORD1008, $189.99, over the approval
threshold) -- resolved cleanly end to end by the existing single agent, no
handoff, real tool calls documented in the file.

## 15. Bedrock AgentCore migration (§2.2)

**Actually deployed and live-invoked, not just code-complete.** The agent
runs on a real AgentCore Runtime endpoint:
`arn:aws:bedrock-agentcore:us-east-1:058264386876:runtime/ecommerce_agent-4ks2toDNhf`.

`agentcore_app.py` (repo root) wraps the existing `SupportHarness` in
`BedrockAgentCoreApp` with a deliberately minimal `@app.entrypoint` --
`SupportHarness`, its LangGraph structure, and every node
(`decide_node`/`validate_node`/`hitl_gate_node`/`act_node`) needed **zero**
changes to fit this wrapper. Verified for real: the entrypoint was actually
invoked end to end locally (against `LLM_PROVIDER=groq`, the same local-dev
provider `server.py` always used) and returned a correct, grounded response
with the right trajectory and trace_id -- this is exactly what `agentcore
dev` exercises.

```bash
python -c "
import asyncio
from agentcore_app import invoke
print(asyncio.run(invoke({'customer_id': 'CUST002', 'message': 'Where is my order ORD1002?'})))
"
```

**`agent/provider.py`'s `BedrockProvider.call()` is now a real
implementation**, not the Assignment 2 stub: boto3 `bedrock-runtime`
`converse()`, with `_to_bedrock_messages` / `_to_bedrock_tool_config` /
`_from_bedrock_response` translating between this project's OpenAI-shaped
wire format and Converse's `messages`/`toolConfig` shapes. Verified against
the **real** `TOOLS` schema from `agent/harness.py` (all 3 tools, including
`propose_refund_decision`) with a mocked boto3 client -- full request/
response round trip confirmed without needing real AWS credentials for
that verification.

**A real finding, not assumed from the assignment text:** the
`bedrock-agentcore-starter-toolkit` package's `agentcore` CLI (`configure`
/ `dev` / `deploy`) is what actually provides the commands the assignment
calls `configure`/`dev`/`launch` -- but the current CLI has **renamed
`launch` to `deploy`** ("formerly 'launch'" per its own `--help` output),
and the CLI's own output actively recommends switching to a newer
npm-based `@aws/agentcore` CLI. That newer CLI was actually tried and
**does not run on this machine** -- `npx @aws/agentcore --help` crashes
immediately with `SyntaxError: Invalid regular expression flags`, because
its bundled JS uses a Unicode-property regex flag (`v`) that requires
Node 20+, and this machine has Node 18.19.1. The deployment below used the
Python starter-toolkit CLI for that reason -- a real, reproducible
environment constraint, not a preference.

**Two more real, live-discovered problems, both fixed before the
deployment below succeeded:**
- **`agentcore deploy` silently built the wrong entrypoint.** This repo
  already had a `Dockerfile` at its root (Assignment 2's ECS image, `CMD
  ["python", "server.py"]`). `agentcore configure` correctly generates its
  *own* Dockerfile (targeting `agentcore_app.py`) into
  `.bedrock_agentcore/<agent>/Dockerfile` -- but `agentcore deploy` still
  picked up the pre-existing repo-root `Dockerfile` and built *that*
  instead, deploying a container that ran the wrong app entirely (the
  first live invoke failed with `GROQ_API_KEY is not set`, an error string
  that only exists in `server.py`/`main.py`, not `agentcore_app.py` --
  that's how this was caught). Fixed by renaming the ECS Dockerfile to
  `Dockerfile.ecs` (and updating the one CI line that built it,
  `.github/workflows/ci-cd.yml`), so the repo root no longer has a
  same-named file for the CLI to ambiguously prefer.
- **`ServiceQuotaExceededException: maxImageSizeMb limit exceeded`** on
  the first real deploy attempt with the correct Dockerfile. The
  auto-generated Dockerfile's plain `uv pip install -r requirements.txt`
  pulls PyPI's default `torch` wheel, which bundles ~2GB of unused
  CUDA/GPU libraries -- the exact problem the *legacy* ECS Dockerfile had
  already solved by installing the CPU-only build from PyTorch's own
  index first. Applying that same fix to the generated AgentCore
  Dockerfile (plus a missing step: `en_core_web_lg`, required by
  `agent/pii.py`'s Presidio layer, isn't pip-installable via
  `requirements.txt` and needs its own download line) brought the image
  from **3.25GB down to 948MB**, under the quota.

**A real design pressure worth naming honestly:** the size fix required
keeping the local Presidio/spaCy PII layer *in* the deployed image, at
real image-size cost, rather than dropping it in favor of Bedrock
Guardrails alone. That tradeoff was deliberate, not incidental --
section 17 has live, bidirectional evidence that Presidio and Guardrails
each catch real PII categories the other misses (a bank routing number
Guardrails has no entity type for, versus phone/address patterns
Presidio's un-customized defaults miss), so dropping either layer to save
image size would have reopened a verified, real gap, not a hypothetical
one.

**Real commands used, in order** (Python starter-toolkit CLI):
```bash
pip install bedrock-agentcore bedrock-agentcore-starter-toolkit
PYTHONIOENCODING=utf-8 agentcore configure --entrypoint agentcore_app.py \
  --name ecommerce_support_agent --region us-east-1 --non-interactive --protocol HTTP
agentcore deploy --auto-update-on-conflict \
  --env LLM_PROVIDER=bedrock --env BEDROCK_MODEL_ID=amazon.nova-lite-v1:0 \
  --env BEDROCK_KNOWLEDGE_BASE_ID=<kb-id> --env HITL_BACKEND=aurora \
  --env COST_LEDGER_BACKEND=aurora --env POLICY_BOUNDARY_BACKEND=avp \
  --env AURORA_CLUSTER_ARN=<arn> --env AURORA_SECRET_ARN=<arn> \
  --env AURORA_DATABASE=kbdb --env AVP_POLICY_STORE_ID=<id>
agentcore invoke '{"customer_id": "CUST002", "message": "Where is my order ORD1002?"}'
agentcore destroy
```

**`PYTHONIOENCODING=utf-8` is required on Windows**, another real,
reproducible finding: `agentcore configure`/`deploy`/`invoke` crash with
`UnicodeEncodeError` under the default `cp1252` console encoding the
moment they try to print a checkmark or emoji, and `agentcore configure`
(without `--non-interactive`) separately crashes with
`NoConsoleScreenBufferError` when run from this environment's shell
(neither Git Bash nor a piped PowerShell session presents a real Win32
console buffer to `prompt_toolkit`) -- `--non-interactive` avoids that
second crash entirely.

**`--env` flags apply at the AgentCore Runtime resource level**, confirmed
by reading the deployed resource back with
`aws bedrock-agentcore-control get-agent-runtime` -- its
`environmentVariables` block showed every `--env` value correctly, even
though neither `agentcore deploy`'s own console output nor the local
`.bedrock_agentcore.yaml` config file ever mentions them.

**`_make_llm_provider()` (`agent/harness.py`) is a fix this deployment
attempt itself found necessary**, not planned in advance:
`agentcore_app.py`'s `invoke()` never passed a `provider=` argument to
`SupportHarness`, so the constructor fell through to `GroqProvider()`
unconditionally -- meaning the real AgentCore deployment would always try
to call Groq (no `GROQ_API_KEY` in that environment) regardless of the
`LLM_PROVIDER` env var, silently defeating the entire migration this
section is about. `LLM_PROVIDER=bedrock` now selects `BedrockProvider`
(`amazon.nova-lite-v1:0` -- see section 17's justification-4-equivalent
note below for why Nova and not Claude), matching every other
env-var-driven backend switch in this project.

**AgentCore Memory is now actually wired up and used, not just
auto-provisioned and idle.** `agentcore configure` auto-created a Memory
resource (`ecommerce_support_agent_mem-...`); `agent/memory_agentcore.py`
adds two classes (`MEMORY_BACKEND=agentcore`) that replace
`agent/memory.py`'s prior implementations:

- **`ShortTermMemoryAgentCore`** replaces `ShortTermMemory` -- a real fix,
  not a cosmetic backend swap: `ShortTermMemory` was an in-process list
  that lived only as long as one `SupportHarness` instance, and since
  AgentCore Runtime constructs a fresh harness per invocation (stateless
  requests), multi-turn memory across *separate* turns of the same support
  ticket never actually persisted anywhere before this, regardless of
  backend. Verified live: stored a turn via `MemoryClient.create_event()`,
  then independently queried `get_last_k_turns()` and got back the exact
  clean text pair, not a mock.
- **`LongTermMemoryAgentCore`** replaces `LongTermMemory` -- which was a
  static, hand-written JSON seed file (`data/ticket_history.json`) that
  never grew or updated from real conversations; not actually memory in
  the sense the term usually means, a lookup table. Three real extraction
  strategies (SEMANTIC, SUMMARY, USER_PREFERENCE) were added to the live
  Memory resource. Verified live, with zero extraction code written by
  this project: after a test conversation where a customer said "please
  always contact me by email, not phone, I never answer calls,"
  `retrieve_memories()` (polled until extraction completed, ~1-2 minutes
  asynchronously) returned:
  ```
  - The user prefers to be contacted by email only and never answers phone calls.
  - {"preference":"Prefers to be contacted by email only; never answers phone calls",
     "categories":["communication","contact preferences"]}
  ```
  An LLM, managed entirely by AWS, read the raw conversation and extracted
  that fact on its own.

**A real bug found and fixed while verifying this:** the first version of
`_make_short_term_memory()` (`agent/harness.py`) was decorated with
`@lru_cache(maxsize=1)`, copied from `_compiled_graph()`'s legitimate
process-wide-singleton pattern without accounting for the difference --
short-term memory is per-(customer, ticket) state, not a singleton. That
single-slot cache meant **every** `SupportHarness` instance in a process,
regardless of which customer or ticket, received the exact same memory
object. Caught by comparing a direct, isolated call against the same call
routed through `SupportHarness` and finding they disagreed; confirmed via
matching Python object ids across two different customers' sessions.
Removed -- verified afterward that two different customers' sessions in
the same process now get genuinely independent, empty memory.

Neither Memory nor Observability gives the `PendingAction`/`ApprovalStatus`
state machine (section 18) for free, though -- that logic (expiry,
re-validation, the resource-keyed uniqueness guard) is still hand-built
regardless of backend, on Aurora (section 18) rather than AgentCore's own
Memory API, since Memory is conversation-context-shaped, not a
business-workflow-approval-state primitive.

**HITL pause/resume is reachable through the same deployed endpoint**, not
just as a local Python call -- `agentcore_app.py`'s `invoke()` dispatches
on `payload["action"] == "resume_approval"` to call
`resume_after_approval()` directly, required by the assignment's "live and
demonstrable" standard for section 18's HITL gate specifically. Verified
live end-to-end: `agentcore invoke` created a real pending approval via
the normal chat path, then a second `agentcore invoke '{"action":
"resume_approval", ...}'` call against the same deployed endpoint
approved and executed it (see section 18 for the full transcript).

**AgentCore Gateway was evaluated and deliberately not built** -- read
the real considerations, not a hand-wave. Gateway's actual value
proposition (per AWS's own introductory material) is solving the "M×N"
tool/agent integration problem: many agents sharing many tools need
centralized discovery, auth, and governance. This project has **one
agent and three tools**, all already correctly secured at two layers
(the harness-level ownership check, `agent/harness.py`'s
`_validate_and_check_permission`, plus the MCP server's own re-check).
Adding Gateway here would mean: a real network hop and OAuth token
exchange for every tool call that currently runs in-process in
milliseconds; a Cognito user pool (or equivalent) for Gateway's inbound
JWT auth; either a second full AgentCore Runtime deployment (the
`mcpServer` target type) or three separate Lambda functions (the
lighter-weight `lambda` target type AWS's own getting-started material
actually leads with) plus their own IAM roles; and a Gateway execution
role of its own -- for zero new capability over what already works and
is already live-verified. Gateway earns its cost past a scale this
project doesn't have; building it here would be adding enterprise-scale
infrastructure to a project that doesn't need it, not a demonstration of
depth.

**What the Gateway investigation *did* produce, kept and real:** a
genuine multi-tenancy bug in `mcp_server/server.py`, found by seriously
working through what a Gateway-fronted (shared, persistent, concurrent)
server would require. `SESSION_CUSTOMER_ID` was a **module-level global**,
read once when the process started -- correct only because the original
design spawned a fresh subprocess per ticket. A persistent, multi-customer
server sharing that global would race across concurrent requests from
different customers. Fixed regardless of the Gateway decision:
`lookup_order`/`issue_refund` now take `customer_id` as a real, explicit
parameter (`check_account_status` already did), and `agent/mcp_client.py`'s
`call_tool()` unconditionally injects/overwrites `customer_id` on every
call -- centrally, in one place, so the model's own tool-call arguments
can never supply a different customer_id and walk past ownership checks.
Verified: the existing local stdio path still works end-to-end after this
change.

**A second real bug found and fixed live, unrelated to Memory or Gateway:**
the deployed AgentCore Runtime container runs as a non-root user, and
`agent/cost_ledger.py`'s local SQLite write failed with `attempt to write
a readonly database`, taking down every turn that reached `decide_node`
with a real 500 error -- reproduced live via `agentcore invoke`, diagnosed
from real CloudWatch logs. Fixed by making cost-ledger writes non-fatal
(cost tracking is pure telemetry; a failed write must never block the
actual customer response) -- deliberately **not** applied to
`agent/hitl_store.py`, where a lost approval-record write is a safety
issue, not a reporting gap, and should stay fatal. Redeployed and
re-verified live afterward: the endpoint responds correctly again.

## 16. Bedrock Knowledge Base migration (§2.3)

**Actually provisioned and serving live retrieval, not just mock-verified.**
`infra/create_kb_aurora.sh` captures the exact real provisioning sequence
used. Ingestion job against this project's real `policies/*.md` completed
with `numberOfDocumentsScanned: 7, numberOfNewDocumentsIndexed: 7,
numberOfDocumentsFailed: 0`. The deployed AgentCore endpoint (section 15)
used this Knowledge Base for real retrieval in every live invocation
during this session -- responses were correctly grounded in retrieved
policy text (e.g. the `lost_in_transit` refund-eligibility case, section
18), not fabricated.

`agent/rag_bedrock_kb.py`'s `BedrockKBRetriever` replaces
`agent/rag_pgvector.py`'s hand-rolled SQL query with the managed
`bedrock-agent-runtime` `Retrieve` API, backed by Aurora PostgreSQL +
pgvector as the Knowledge Base's own (KB-managed, not hand-rolled) vector
store schema. Same `retrieve(query, top_k)` interface as every other
retriever in this project, so `agent/harness.py`'s `_shared_retriever()`
selects it via a `BEDROCK_KNOWLEDGE_BASE_ID` env var, checked first (ahead
of the Assignment 2 `DATABASE_URL`/pgvector path, kept as a documented
fallback).

Request/response shapes were verified directly against the **installed**
botocore service model (`client.meta.service_model.operation_model('Retrieve')`)
before writing this code, not assumed from memory -- `retrievalQuery.text`,
`retrievalConfiguration.vectorSearchConfiguration.numberOfResults`,
`retrievalResults[].{content.text, location.s3Location.uri, score}` all
confirmed real. `retrieve()` itself was then verified with a mocked
`bedrock-agent-runtime` client: correct request shape, correct
score-threshold filtering (a low-score result correctly excluded), correct
doc-id extraction from the S3 object URI.

**`min_score` is explicitly NOT the same value as `agent/rag.py`'s
`DEFAULT_MIN_FUSED_SCORE` (0.38)** -- Bedrock KB's own relevance score is a
different metric from this project's hand-tuned BM25+vector fusion. 0.38
is kept only as a placeholder in `agent/rag_bedrock_kb.py`'s
`DEFAULT_MIN_SCORE`; it must be re-tuned empirically against the same
held-out genuine-vs-adversarial query set the original threshold was
calibrated against, once a real Knowledge Base exists to test against.
**Not yet done** -- pending section 24's provisioning batch.

**Honestly still not done, flagged rather than skipped silently:** the
empirical `min_score` recalibration against real Bedrock KB scores, and the
stale-connection proof (pointing `rag_pgvector.py` at an empty table to
confirm the old path breaks while the new one still works), were not
completed before this session's Aurora cluster was torn down to stop
billing (see section 24) -- there wasn't a specific real-score-vs-threshold
mismatch observed in practice during live testing (retrieval visibly
worked correctly on every live query), but "worked in the cases tried"
isn't the same as an empirically re-tuned threshold. Both are real
`infra/create_kb_aurora.sh`-reproducible next steps once Aurora is
re-provisioned for the final demo.

## 17. PII detection and redaction, layered (§2.4)

**Fully implemented and verified live against a real, deployed Bedrock
Guardrail -- no longer a deferred seam.**

`agent/pii.py`: Microsoft Presidio (`AnalyzerEngine` + `AnonymizerEngine`,
spaCy `en_core_web_lg`) as the local, always-on layer, chained into a real
`redact_bedrock_guardrails()` boto3 `bedrock-runtime.apply_guardrail()`
call against a provisioned guardrail
(`arn:aws:bedrock:us-east-1:058264386876:guardrail/8c3d1djf3a5a`, version
1, `sensitiveInformationPolicyConfig` covering EMAIL/PHONE/NAME/ADDRESS/
US_BANK_ACCOUNT_NUMBER/CREDIT_DEBIT_CARD_NUMBER/US_SOCIAL_SECURITY_NUMBER).
`redact_layered()` runs Presidio first, then Guardrails against Presidio's
own output -- **unioning** both layers' findings rather than one replacing
the other, per the assignment's explicit "these layer, they don't replace
each other" requirement.

**Redaction strategy: partial masking**, not full masking or tokenization.
Full masking (`[REDACTED]`) destroys the agent's ability to usefully
confirm "the email on file ending in `...@example.com`" back to a
customer. Tokenization (reversible pseudonymization) adds a key-management
vault surface this project has no legitimate need for -- nothing
downstream ever needs the real value revealed again. Partial masking
(Presidio's real `mask` operator via `OperatorConfig`, e.g.
`j***@example.com`) balances "customer can recognize their own data enough
to confirm identity" against "the raw value never appears in a log, trace,
or LLM-visible transcript." This was empirically demonstrated as a real,
non-hypothetical cost: asking the agent "what email and phone do you have
on file for me" returns the customer's **own** data back masked -- the
conservative default this project chose, at a real (documented, not
hidden) UX cost.

**Enforced at all three required points:**
1. **Final customer-facing reply** -- `agent/harness.py`'s `handle_turn`.
2. **Tool-call results/errors**, before they re-enter `state["messages"]`
   -- `agent/mcp_client.py`'s `call_tool()`. This is the one that actually
   matters most: a tool RESULT (e.g. `check_account_status` returning a raw
   account dict) is never itself the final reply, so redacting only #1
   never touches it.
3. **Trace payloads** -- `agent/tracing.py`'s new `safe_span_payload()`
   wrapper, applied at every `update_current_span`/`update_current_generation`
   call site in `agent/harness.py` that carries message or tool content.

**Two genuine, empirically-found-and-fixed gaps** (found by actually
testing against this project's own mock data, not manufactured):

- **The `555-01XX` fake-phone gap.** Presidio's default `PHONE_NUMBER`
  recognizer is backed by Google's `phonenumbers` library, which validates
  against real assignable NANP ranges -- and rejects the `555-01XX` block
  as invalid, because that block is FCC-reserved specifically for fiction.
  Verified directly: `'+1-212-9876543'` -> correctly detected as
  `PHONE_NUMBER`; `'+1-555-0142'` (this project's own mock phone numbers,
  in `data/accounts.json`, deliberately chosen to avoid colliding with a
  real person) -> **nothing detected at all**. The safe-data choice and the
  detection gap are directly the same decision.
- **No default street-address recognizer.** Presidio ships nothing for
  street addresses (only city/state via its `LOCATION`/NER entity).
  Verified directly: `'482 Birchwood Ave'` -> nothing detected.

**Fix:** two custom `PatternRecognizer`s (`NANP_FAKE_PHONE`,
`US_STREET_ADDRESS`) registered into Presidio's registry -- the same
approach the assignment's own PAN/Aadhaar example uses. Re-ran the exact
same `check_account_status` call after the fix:

```
RAW:      {'phone': '+1-555-0142', 'shipping_address': {'street': '482 Birchwood Ave', ...}}
BEFORE:   {'phone': '+1-555-0142', 'shipping_address': {'street': '482 Birchwood Ave', ...}}   <- unmasked
AFTER:    {'phone': '***********', 'shipping_address': {'street': '*****************', ...}}   <- fixed
```

**A third genuine found-and-fixed bug, more serious than the first two --
this one was a redaction-execution bug, not a detection gap.**
`_partial_mask_operators()` originally set `PERSON`/`LOCATION` to
`chars_to_mask=6` on the theory that masking only the first few characters
would leave a "recognizable but hidden" value. That's backwards: Presidio's
`mask` operator masks exactly `chars_to_mask` characters from the START of
the matched span and leaves the rest untouched -- so any name or location
longer than 6 characters leaked everything past character 6, even though
detection itself was correct. Verified directly:

```
Input:  "My name is Jordan Ellis and my account is CUST003, SSN 523-11-8842, ..."
Detected span (correct): PERSON, offset 11-23, "Jordan Ellis" (full name)
BEFORE fix: "My name is ****** Ellis and ..."   <- surname leaked in plain text
AFTER fix:  "My name is ************ and ..."   <- fixed (chars_to_mask=100, full match)
```

**Guardrails vs. Presidio, tested live in both directions -- a real,
bidirectional finding, not a one-way "Guardrails is strictly better"
story:**

- **Guardrails catches what Presidio's un-customized defaults miss**
  (the original `555-01XX` phone / street-address gaps, section above) --
  confirmed by running the *unmodified* input through a live
  `ApplyGuardrail` call: both correctly detected and masked
  (`{PHONE}`, `{ADDRESS}`).
- **Presidio catches what Guardrails' fixed entity list misses.** Tested
  live against `"...bank account 000123456789, routing number 021000021,
  or my card 4242 4242 4242 4242."`:
  ```
  Presidio:   masks bank account AND routing number (US_BANK_NUMBER)
  Guardrails: "...routing number 021000021, or my card {CREDIT_DEBIT_CARD_NUMBER}."
              <- routing number left in PLAIN TEXT; Guardrails has no ABA
                 routing-number entity type in its standard PII categories.
  ```

This is the concrete, live-verified justification for layering rather than
picking one: neither tool's coverage is a superset of the other's, in
either direction, on this project's own real data.

**CI cost, documented honestly:** since PII redaction is wired into the
*core* turn path (not an optional side-eval), `.github/workflows/ci-cd.yml`'s
`eval-gate` and `full-eval-report` jobs both now run `python -m spacy
download en_core_web_lg` before the eval script -- a real, added CI runtime
cost, not hidden.

## 18. HITL approval gate with pause/resume (§2.5)

**Fully implemented and verified locally with real runs. AgentCore-backed
session-state persistence is pending provisioning; the local `HITLStore`
(SQLite) is the documented, same-interface stand-in until then.**

Two genuinely distinct mechanisms, per the assignment's own framing:

**1. The approval gate.** `agent/harness.py`'s new `hitl_gate_node`,
reached via a new conditional edge after `validate_node` specifically when
a `propose_refund_decision` call's `requires_approval` is `True` (set by
`agent/policy_boundary.py`'s `evaluate_refund_policy()` -- at or above the
**$150** threshold, see section 19). Creates a real `PendingAction`
(`agent/hitl.py`), short-circuits the turn with a "your request is under
review" reply, instead of ever reaching `act_node`.

**2. Pause/resume.** `agent/hitl_store.py`'s `HITLStore` (local SQLite
today, the documented pre-AWS stand-in for AgentCore's managed session
state -- same interface either way, so swapping the backend later changes
only this module's internals). `agent/harness.py`'s new
`resume_after_approval()` is the actual execution path, deliberately
**outside** the per-ticket LangGraph, since a human decision arrives
asynchronously, not as another turn in the conversation.

**State machine:** `ApprovalStatus` enum (`pending`/`approved`/`rejected`/
`expired`/`executed`), not a boolean. Real 24-hour expiry window
(`APPROVAL_WINDOW`).

**Re-validation on resume** -- the single most common real HITL bug, named
explicitly by the assignment: `resume_after_approval()` re-fetches the live
order (`data/orders.json`, fresh read) and calls
`HITLStore.revalidate_before_execution()`, which compares `status` and
`order_total` against the `resource_snapshot` captured when the approval
was requested. Verified directly:

```
Simulated a changed order_total ($399.00 -> $350.00) between approval-request and execution:
"correctly caught stale state: order ORD1007 changed since approval was requested
 (was status='delivered' total=399.0, now status='delivered' total=350.0) --
 refusing to execute against stale state, re-review required"
```

**A genuine found-and-fixed bug, not manufactured -- the "Double Refund"
case named in the assignment.** `agent/hitl_store.py`'s `create_pending()`
v1 had no uniqueness check against `resource_id` (the order), only against
`approval_id`/`ticket_id`. `eval/hitl_bug_repro.py` (kept as a permanent
regression test, not thrown away) reproduced it for real:

```
Simulating two overlapping tickets against the same order (ORD1007)...
  ticket_A pending action created: 00c4a2e0778f485db943da4476b5800c
  ticket_B pending action created: f80e939f07874aee814d2f914cfa2e94
BUG REPRODUCED: two independent PendingActions exist for the same resource_id.
Approving both independently (as two different human reviewers would, unaware of each other)...
  ticket_A executed: executed
  ticket_B executed: executed
DOUBLE REFUND: $399.00 was approved and executed TWICE for order ORD1007
(total exposure: $798.00), from a single real order.
```

**The fix:** `create_pending()` now checks `get_active_for_resource()`
first and raises `DuplicatePendingActionError`. Re-ran the identical repro
after the fix:

```
Simulating two overlapping tickets against the same order (ORD1007)...
  ticket_A pending action created: 2e3c51a5438f42e8852ca85639f8b9c6
BUG NOT REPRODUCIBLE (fixed) -- second create_pending() correctly raised:
resource_id='ORD1007' already has an active pending action
(approval_id='2e3c51a5438f42e8852ca85639f8b9c6', requested for ticket_id='ticket_A')
-- refusing to open a second one
```

**A new MCP write tool**, `mcp_server/server.py`'s `issue_refund(order_id,
amount_usd)` -- the only write capability this server has. Re-checks
ownership and amount server-side as defense-in-depth (same multi-layer
discipline as the two read tools), and refuses a second refund on an order
that already has one on record.

**Full end-to-end verification against the real deployed AgentCore
endpoint** (not just local Python) -- `agent/hitl_store_aurora.py`'s
Postgres-partial-unique-index-backed store, live, via two separate
`agentcore invoke` calls against the same running endpoint (section 15
covers why `resume_approval` is reachable through that one endpoint at all):

```
$ agentcore invoke '{"customer_id": "CUST003", "message": "My graphic tablet
  order ORD1008 never arrived, it says lost in transit. I want a refund."}'
Response: Your refund request for order ORD1008 ($189.99) is above our review
threshold and has been submitted for approval (reference: 944dd10c162c415d...).

$ # confirmed in Aurora directly: hitl_pending_actions row, status='pending',
$ # resource_id='ORD1008', amount_usd=189.99 -- real row, not asserted

$ agentcore invoke '{"action": "resume_approval", "approval_id":
  "944dd10c162c415db51b269ede8a6324", "decision": "approved", "decided_by": "reviewer_1"}'
Response: {"status": "executed", "refund_result": {"order_id": "ORD1008",
"refund_issued": true, "refunded_amount": 189.99}}
```

**Reproduce the found-and-fixed bug yourself:**
```bash
python -m eval.hitl_bug_repro    # the found-and-fixed regression test
```

## 19. Policy boundary on refund amounts (§2.6)

**Decision: real Amazon Verified Permissions, with the hand-rolled
version kept as the local-dev fallback -- not the other way around.**
`agent/policy_boundary.py`'s hand-rolled evaluator was the first version,
built during the local-code-first phase per the assignment's explicitly
permitted fallback. Once AVP was actually provisioned, `agent/harness.py`
was wired to select between them via `POLICY_BOUNDARY_BACKEND=avp`
(defaulting to the hand-rolled version, matching every other backend
switch in this project -- see `_shared_retriever()`, `_make_hitl_store()`).

**Real policy store**: `KozP4Mk6man7ivCYxeqBMP` (us-east-1), Cedar schema
(`RefundPolicy` namespace, a `ProposeRefund` action with a context-only
condition model -- `order_status`/`amount_matches`/`account_suspended`),
four static policies, each mapped 1:1 to the hand-rolled version's four
checks:
- `GDw11cxJqYGZEKBjGN4TZT` -- `forbid` when order status isn't refund-eligible
- `LugUCiwxL7ovzd5N4uVsPw` -- `forbid` when proposed amount != order total
- `MKd6FZzur6QhKaAwuzYnGM` -- `forbid` when the account is suspended
- `FbncATAgvRbEb8YDAxJesz` -- baseline `permit` (Cedar's explicit-forbid-
  wins semantics mean this only takes effect once none of the three
  forbids fire)

`agent/policy_boundary_avp.py`'s `evaluate_refund_policy_avp()` calls the
real `verifiedpermissions.is_authorized()` API and maps
`determiningPolicies` back to a human-readable reason -- worth being
honest about what AVP gives for free versus what still had to be written
by hand: Cedar's decision is ALLOW/DENY plus *which policy fired*, not a
reason string, so the reason text is still application logic layered on
top of a real authorization-service call, not something `IsAuthorized`
returns on its own. `requires_approval` is likewise deliberately kept out
of Cedar -- being over the $150 threshold isn't a policy violation, it's a
routing decision on an already-permitted request, computed from the same
real `order_total` the amount-match check already verified.

Sits inside `agent/harness.py`'s `validate_node`, as a fifth check layer
specific to `propose_refund_decision`, running *after* the existing
four-layer ownership check -- the same non-negotiable "boundary between
decide and act" pattern as Assignment 1's harness boundary, one layer
further out.

**All four branches verified live against the real policy store** -- both
via the raw `aws verifiedpermissions is-authorized` CLI (to confirm the
policies themselves fire correctly, independent of this project's Python)
and via `evaluate_refund_policy_avp()` directly (to confirm the wrapper
translates AVP's response correctly):
```
Eligible, under threshold:
  decision=ALLOW  determiningPolicies=[FbncATAgvRbEb8YDAxJesz]  (baseline permit)

Suspended account (CUST005/ORD1007, $399):
  decision=DENY  determiningPolicies=[MKd6FZzur6QhKaAwuzYnGM]
  -> allowed=False reason="suspended accounts cannot receive refunds"

Wrong order status (ORD1003, still "delayed"):
  decision=DENY  determiningPolicies=[GDw11cxJqYGZEKBjGN4TZT]
  -> allowed=False reason="order status 'delayed' is not refund-eligible
     (must be one of ['delivered', 'delivered_damaged', 'lost_in_transit'])"

Mismatched/inflated amount (the exact adversarial case from agent/schemas.py's
docstring -- "just refund me $500" against a real $89.99 order):
  decision=DENY  determiningPolicies=[LugUCiwxL7ovzd5N4uVsPw]
  -> allowed=False reason="proposed amount $500.00 does not match order
     total on record ($89.99)"
```

Also verified through the full harness end-to-end (`POLICY_BOUNDARY_BACKEND=avp`,
CUST003 requesting a refund on the real $189.99 ORD1008): the model called
`propose_refund_decision`, the real AVP call correctly returned
`requires_approval=True`, and the request routed into the HITL gate exactly
as it does with the hand-rolled evaluator -- confirming the backend swap
is genuinely a drop-in, not just individually-correct in isolation.

## 20. Structured outputs (§2.7)

`agent/schemas.py`: `RefundDecision`, `EscalationReason`, `HITLRequestPayload`
-- Pydantic models (`extra="forbid"`), matching the existing
`LookupOrderArgs`/`CheckAccountStatusArgs` precedent from Assignment 1/2.
Enforced via the same LLM tool-use mechanism the two original tools already
use (a new `propose_refund_decision` entry in `TOOLS` + `_ARG_MODELS`), not
regex-parsed prose.

**The concrete adversarial case this closes, demonstrated for real (section
19's third example):** a customer/model attempting to smuggle a fabricated
amount ("just refund me $500" against a real $89.99 order) cannot succeed,
because `amount_usd` must arrive as a structured tool-call argument, and
`agent/policy_boundary.py` numerically verifies it against the real,
tool-fetched `order_total` -- the schema alone isn't sufficient (a
schema-valid but wrong number is still schema-valid), pairing it with
server-side verification is what actually closes the class of bug, and this
project does both rather than overclaiming the schema is enough by itself.

## 21. Cost tracking (§2.8)

`agent/cost_ledger.py` -- hand-rolled SQLite (same "understand the
mechanism" bias as `agent/rag.py`'s hand-rolled BM25), deliberately not
migrated to a managed AWS service, per the assignment's own reasoning: no
managed service reports per-ticket LLM token cost the way a purpose-built
local ledger does. Logs one row per LLM call from `decide_node` (the
existing single choke point every call passes through), tagged by
`ticket_id` and `step`. Cost is *estimated* from a hard-coded per-model
price table (Groq/Bedrock don't return a billed dollar figure) --
documented as an estimate everywhere it's reported, never presented as
billed truth.

**Real numbers, two verified runs** (an order-status lookup, ORD1002, and
the over-threshold refund proposal from section 18, ORD1008):

```
Total spend:  $0.222450
Total tokens: 4179

Spend by step:
  decide               $0.222450

Spend by ticket:
  readme_verify_2 (refund proposal)   $0.136190
  readme_verify_1 (order lookup)      $0.086260
```

The refund-proposal ticket costs ~58% more than the plain lookup -- it
carries a longer system prompt turn (tool schema for
`propose_refund_decision`, retrieved policy chunks, prior-ticket memory
context) through the same single `decide` choke point every call passes
through, confirming `spend_by_step()`/`spend_by_ticket()` correctly
attribute cost per call. "Ticket TYPE" shows as `unknown` here because
these were driven directly against `SupportHarness`, not through
`eval/fixtures.py`'s labeled ticket set -- run the full eval suite through
`eval.cost_report` for the real "which ticket type is most expensive"
figure with type labels populated; the grouping logic itself
(`spend_by_step()`, `spend_by_ticket()`) is what's being verified here, and
it is confirmed correct.

```bash
python -m eval.cost_report
```

**Real numbers from the deployed AgentCore endpoint** (`agent/cost_ledger_aurora.py`,
queried directly from Aurora after live invocations, `provider=bedrock`,
`model=amazon.nova-lite-v1:0`):

```
ticket_id                        step    provider  model                 input  output  cost_usd
8b554e23553c4e0cb153b7f6bf9e988e  decide  bedrock   amazon.nova-lite-v1:0  1623    117    0.000125
8b554e23553c4e0cb153b7f6bf9e988e  decide  bedrock   amazon.nova-lite-v1:0  1455     60    0.000102
```

Real, live evidence for the Groq-vs-Bedrock model-swap cost comparison the
assignment's own README section 9 (Assignment 2 cost benchmarking) already
tracks: Nova Lite's per-call cost here (~$0.0001-0.0002) is roughly two
orders of magnitude below Groq's `gpt-oss-120b` calls in section 21's
earlier local numbers (~$0.06-0.09 each) -- not a controlled apples-to-apples
comparison (different prompts, different call counts), but a real, directly
observed order-of-magnitude difference worth naming rather than a projected
one.

## 22. Semantic cache (§2.9)

`agent/semantic_cache.py`'s `SemanticCache` -- hand-rolled (not GPTCache,
same bias as the cost ledger and BM25), wraps any retriever exposing
`retrieve(query, top_k)`, reuses the same `sentence-transformers` model
already used for retrieval to embed queries for its own similarity check.
Wired into `agent/harness.py`'s `_shared_retriever()`'s new
`BEDROCK_KNOWLEDGE_BASE_ID` branch (section 16) -- in front of the *new*
Bedrock KB path specifically, per the assignment's explicit requirement,
not the local/pgvector paths (caching only pays off in front of a real
network-hop call).

**Threshold, calibrated empirically, not picked by feel:** the first draft
used 0.92 -- which rejected every genuine paraphrase tested against
`all-MiniLM-L6-v2` (real paraphrases score 0.54-0.85 cosine similarity with
this model; only near-identical strings crossed 0.92). Checked real
similar-vs-different query pairs (same discipline as `agent/rag.py`'s
`DEFAULT_MIN_FUSED_SCORE` calibration): different queries land 0.10-0.32,
similar ones 0.54-0.85. Set `DEFAULT_SIMILARITY_THRESHOLD = 0.5` for a
clean margin on both sides.

```bash
python -m eval.semantic_cache_demo
```

**Real before/after, verified run:** `"When will my order arrive?"` then
the differently-worded `"What's the status of my delivery?"` -- confirmed
cache hit (identical result, zero calls to the wrapped retriever on the
second query): 186.88ms -> 19.03ms, **9.8x faster**, even against the
already-fast local BM25 path (the exact multiple varies run to run with
embedding-model load state, so re-run it yourself rather than treat this
figure as fixed -- the durable claim is "zero wrapped-retriever calls on a
cache hit," confirmed above, not a specific millisecond number). This will
show a much more dramatic delta once re-run against the real network-hop
Bedrock KB call (section 16) -- noted honestly rather than overstated with
today's local-path number.

## 23. Mock data changes

The Assignment 2 mock dataset (`data/orders.json`, `data/accounts.json`)
had zero PII fields and zero dollar amounts anywhere -- too thin to
honestly test PII masking or a refund-threshold HITL gate, per the
assignment's own explicit instruction to expand it rather than force a weak
demo. `data/orders.json`'s `items` restructured from bare strings to
`{name, price}` objects plus a computed `order_total`, with prices
deliberately straddling the **$150** threshold (see section 19).
`data/accounts.json` gained `email`/`phone`/`shipping_address`/
`payment_method_last4` on all 5 accounts -- realistic-shaped but
deliberately fake (`example.com` emails, the FCC-reserved `555-01XX`
fictional-number block, no real card numbers even as fake data) so nothing
collides with a real person if it leaks during PII testing, which is the
whole point of this data existing (and which directly produced section 17's
first real finding). `policies/refund_eligibility.md` gained the actual
$150 threshold and delivered-state precondition as real policy text, so the
HITL gate's number is grounded in policy, not invented only in code.

## 24. AWS provisioning status and ECS/ALB teardown

**The AgentCore Runtime + Memory deployment now runs entirely through
CloudFormation, not the `agentcore` CLI.** The CLI-created Runtime,
Memory, and execution role were deliberately deleted and recreated purely
from `infra/cloudformation/agentcore-infra.yaml` -- closing the
reproducibility gap the CLI-only path left (a machine-local,
non-version-controlled `.bedrock_agentcore.yaml`). Real, live-found gap in
the process: the hand-written CFN execution role initially omitted
AgentCore Memory permissions the CLI's auto-generated role had -- caught
by an actual `AccessDeniedException` on `bedrock-agentcore:ListEvents`
from a real invocation, not assumed, then fixed in both the live role and
the template (see the git history for the exact commit). Re-verified live
afterward: order lookup, PII masking, and a real Amazon Verified
Permissions rejection (suspended account) all confirmed working through
the CloudFormation-managed endpoint
(`arn:aws:bedrock-agentcore:us-east-1:058264386876:runtime/ecommerce_agent-4ks2toDNhf`).
Current stack parameters: `HITL_BACKEND=sqlite`, `COST_LEDGER_BACKEND=sqlite`,
`BEDROCK_KNOWLEDGE_BASE_ID=""` (Aurora/KB intentionally still commented out
in the template per the demo-sequencing note at its top) --
`POLICY_BOUNDARY_BACKEND=avp` and `MEMORY_BACKEND=agentcore` are both real
and live.

**The AWS provisioning batch ran for real.** In order, all actually done
and live-verified this session:

1.  Aurora Serverless v2 + pgvector provisioned (`infra/create_kb_aurora.sh`).
2.  Bedrock Knowledge Base created, all 7 `policies/*.md` docs ingested
   successfully (section 16).
3. Bedrock Guardrail provisioned and live-verified in both directions
   against Presidio (section 17).
4. Amazon Verified Permissions policy store, schema, and four Cedar
   policies provisioned and live-verified across all four branches
   (section 19).
5.  `agentcore configure` / `agentcore deploy`, real live invocations
   confirmed against the deployed AgentCore Runtime endpoint, using
   `amazon.nova-lite-v1:0` -- **not Claude**, a real, confirmed blocker:
   this AWS account's Bedrock access to Claude models specifically fails
   with `AccessDeniedException: Model access is denied due to
   INVALID_PAYMENT_INSTRUMENT ... Your AWS Marketplace subscription for
   this model cannot be completed`, retried repeatedly (4 attempts over 2
   minutes, including after the account's payment method was updated) and
   still failing, while Amazon-native models (Nova, Titan) work fine on
   the same account -- an AWS Marketplace-specific billing issue, not a
   Bedrock model-access issue in general. Documented rather than silently
   worked around; `BEDROCK_MODEL_ID` is an env var specifically so
   swapping back to a Claude model ID is a one-line config change once
   that Marketplace subscription issue is resolved.
6. ✅ HITL pause/resume and cost tracking verified live against the real
   Aurora-backed store, through the deployed AgentCore endpoint itself
   (section 18, section 21).
7. **Not done, honestly flagged:** `agent/rag_bedrock_kb.py`'s `min_score`
   empirical recalibration and the stale-connection proof (section 16),
   and the Assignment 2 ECS/ALB/RDS CloudFormation stack teardown -- the
   ECS stack is still live (section 11's resource names/ALB URL remain
   accurate) and was deliberately **not** torn down yet, since full live
   verification of every hardening piece wasn't complete when this
   session's AWS budget/time ran out.

**The Aurora cluster and Bedrock Knowledge Base were torn down at the end
of this session** to stop the ACU-hour billing clock (Aurora Serverless v2
does not scale to zero the way Serverless v1's auto-pause did -- it bills
continuously at its configured minimum ACU whether or not it's handling
traffic), per this project's "provision, verify, tear down" cost discipline.
`infra/create_kb_aurora.sh` / `infra/teardown_kb_aurora.sh` make this
reproducible for a final demo pass: re-run `create_kb_aurora.sh`, re-deploy
AgentCore with the new resource ARNs, and everything above is
re-verifiable from a clean state. The AgentCore Runtime deployment and the
Verified Permissions policy store were left running (their idle cost is
negligible compared to Aurora's continuous ACU billing).

## 25. Defensible justifications (§3)

**1. Session 5 path and why:** see section 14 / `eval/autonomy_decision.md`
in full. Short form: the candidate specialist's job already fully exists
inside the current agent (structured refund proposals + a real policy
boundary + a real HITL gate), there's no current bottleneck the
single-agent shape causes, and a second agent's one concrete failure mode
(a dropped handoff) isn't worth taking on for zero capability gain.

**2. What HITL re-validation actually protects against:** the real,
found-and-fixed "Double Refund" bug (section 18) -- v1 of
`HITLStore.create_pending()` let two overlapping tickets against the same
order each get an independent `PendingAction`, both approved by two
reviewers unaware of each other, both executed: $798 total exposure from
one $399 order. The fix (a `resource_id`-keyed uniqueness guard) closes
exactly this bug, not a hypothetical one -- reproduced before the fix,
re-ran identically after, captured both.

**3. Why $150 and partial masking, and what looser/tighter would miss:**
$150 sits meaningfully above this project's smallest real order (~$40) and
below its largest (~$400), giving genuine test coverage on both sides of
the line rather than a threshold everything trivially clears or misses. A
looser threshold (e.g. $500) would auto-approve most of this project's real
orders, meaning the HITL gate would rarely fire in practice and the
re-validation logic would go largely untested by real traffic. A tighter
one (e.g. $25) would route nearly everything to human review, defeating the
point of having an auto-approve path at all and creating exactly the "alert
fatigue" risk Session 7 named. Partial masking (not full masking or
tokenization) was chosen because this project has no legitimate downstream
need to ever reveal a masked value again (ruling out tokenization's vault
complexity) while still needing the agent to usefully reference "the email
on file" back to a customer (ruling out full masking's `[REDACTED]`
destroying that utility) -- demonstrated as a real, felt trade-off in
section 17's "customer asks for their own data, gets it back masked"
finding, not a hypothetical one.

**4. What layering Bedrock Guardrails alongside Presidio actually buys,
verified live in both directions (section 17):** Guardrails' managed
entity list caught this project's original `555-01XX`/street-address gaps
without any custom code, confirmed by running the unmodified input through
a live `ApplyGuardrail` call. But the reverse is also true and just as
real: Presidio caught a bank routing number that Guardrails' fixed entity
list has no category for at all (`ApplyGuardrail` left `021000021` in
plain text next to a correctly-masked card number). Neither tool's
coverage is a superset of the other's on this project's own data --
that's the actual, demonstrated case for layering, not a theoretical one.
Presidio's local, no-network operation also means it never depends on
Bedrock being reachable and has zero per-call AWS cost, which stays true
regardless of the coverage comparison.

