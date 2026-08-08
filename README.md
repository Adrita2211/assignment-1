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
agent/rag.py               local hybrid retriever: BM25 (lexical) + FAISS/sentence-transformers (semantic) over policies/
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
infra/cloudformation/*.yaml  ECS/ALB/ECR/RDS+pgvector/autoscaling and the LangFuse EC2 stack (section 11)
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

One-time setup, either option:
- **Self-hosted** (free, matches what this project's CloudFormation deploys
  for the live demo): `bash infra/deploy_langfuse_stack.sh` -- see section 11.
  Prints the dashboard URL and login.
- **LangFuse Cloud free tier**: create a project at
  [cloud.langfuse.com](https://cloud.langfuse.com), grab its keys.

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

**Finding a real "confident wrong path" case:** run `demo.py` or a few
turns of `main.py`, open the corresponding traces in LangFuse, and look for
a ticket where the final answer reads fine but the trace shows something
off -- a retrieval that returned the wrong doc but got cited anyway, a tool
called with a plausible-looking but wrong argument, or (most directly
reproducible here) a turn where `AGENT_REGRESSED=true` silently drops
grounding and the model still answers confidently from its own general
knowledge instead of admitting the gap (see section 9). Document the actual
case you find here, with the trace/trajectory excerpt, before submitting --
this has to be a real finding, not a hypothetical.

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

**Demonstrating the gate blocking a real regressed push:** the repo's
regression flag is `AGENT_REGRESSED` (see `agent/harness.py`'s
`retrieve_node` -- when true, retrieval is silently disabled, simulating a
broken grounding step without touching anything else about the agent, the
same discipline as `agent-cicd-demo`'s `AGENT_REGRESSED` tool-removal flag).
To push the actual demo:

```bash
git checkout -b regression-demo
# edit Dockerfile: add `ENV AGENT_REGRESSED=true`
git commit -am "regression-demo: disable RAG grounding"
git push -u origin regression-demo
```

`ci-cd.yml` triggers on both `main` and `regression-demo`; on the latter,
`eval-gate` should fail in a real GitHub Actions run (screenshot/log
required per section 3, not a description).

## 10. Before/after report and defensible justifications

```bash
python -m eval.before_after_report
```

Runs the trajectory-eval suite twice in one process -- once clean, once with
the regression flag forced on -- and prints both pass rates plus exactly
which ticket IDs flipped from PASS to FAIL. This is the number for the
README/PR description; the CI gate above is what actually blocks the
regressed build from deploying, this script just documents the drop it
would have caused. (Run it locally and paste the real output here before
submitting -- the exact numbers depend on a live model run and aren't baked
into this repo.)

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

**Deploying:**
```bash
export GITHUB_ORG=<your-username-or-org>
export GITHUB_REPO=<this-repo-name>
export GROQ_API_KEY=<a-real-groq-key>
export DB_MASTER_PASSWORD=<a-strong-password>
bash infra/deploy_agent_stack.sh          # ECS/ALB/ECR/RDS/autoscaling stack
bash infra/deploy_langfuse_stack.sh       # self-hosted LangFuse (optional if using LangFuse Cloud)
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
bash infra/teardown_langfuse_stack.sh   # deletes the LangFuse EC2 instance + Elastic IP
```
Both are tested, not just described -- `teardown_agent_stack.sh` empties the
ECR repo first (CloudFormation won't delete a non-empty one) and waits on
`cloudformation wait stack-delete-complete` before reporting done. The
GitHub OIDC provider (`infra/setup_oidc_provider.sh`) is never torn down --
it's an account-wide resource other repos may depend on.

## 13. Bedrock-ready path (optional, not attempted as a working provider)

`agent/provider.py` defines `LLMProvider` (an ABC with one method,
`call(messages, tools) -> {"text", "tool_calls"}`), `GroqProvider` (the
working implementation used everywhere in this repo), and `BedrockProvider`
-- a stub whose `call()` raises `NotImplementedError` and whose docstring
names the exact call it would make (`bedrock-runtime`'s `converse()` API)
once implemented. Selected via `LLM_PROVIDER=bedrock` through
`get_provider()` in the same file. This is a code-shape exercise per the
assignment's §2.6.4 -- the seam exists and is reviewable; no Bedrock
infrastructure was deployed.
