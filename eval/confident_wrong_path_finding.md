# Confident Wrong Path: a real finding (Assignment 2, §2.2)

This is a real case found by inspecting an actual Langfuse trace against the
deployed agent, not a fixture written to satisfy the requirement.

## The ticket

- **Customer:** CUST005 (Elena Petrova)
- **Message:** *"My account is under review and I want to stop being charged
  every month, what do I do?"*
- **Trace ID (Langfuse):** `119d9e059b6127e0e81673297fab3327`

## The final answer (looks fine on its own)

> Hi Elena,
>
> Your account is currently **suspended** (see policy
> [#account_suspension_appeal]). While a suspended account can't place new
> orders, you can still manage your subscription.
>
> To stop the monthly Plus charge:
>
> 1. Log in to your account on our website.
> 2. Go to **Account → Subscriptions**.
> 3. Select the **Plus (monthly)** plan and choose **Cancel subscription**.
> 4. Confirm the cancellation.
>
> Cancelling will prevent any further recurring charges. If you'd like to
> have the suspension reviewed and potentially reactivate your account, you
> can submit an appeal by escalating this ticket to the **account-review**
> queue; the appeal will be reviewed within 5 business days.

Read on its own, this is a confident, specific, professional answer. A
final-answer-only check (does it mention cancellation, is the tone right, is
there a clear next step) would pass this without hesitation.

## What the trajectory actually reveals

The `retrieve_policy` span's output for this trace:

```
retrieved_doc_ids: ['account_suspension_appeal']
```

Only `account_suspension_appeal.md` was retrieved. `subscription_cancellation.md`
was never retrieved at all -- the query blends two topics ("account under
review" -> suspension; "stop being charged" -> cancellation), and only the
suspension doc cleared the retrieval threshold.

Checked both policy docs directly:

- `policies/account_suspension_appeal.md` (the only doc actually retrieved)
  contains nothing about a cancellation UI flow.
- `policies/subscription_cancellation.md` (never retrieved) covers only
  billing/refund rules -- no partial refund on monthly plans, prorated
  refund for annual plans cancelled within 30 days -- and contains **zero**
  UI navigation steps.

**The 4-step cancellation procedure in the final answer is entirely
fabricated.** It isn't grounded in the doc that was retrieved, and it isn't
grounded in the doc that would have actually been relevant (which was never
retrieved). The model filled the gap from its own general knowledge of
e-commerce UIs.

## Why this is a genuine confident-wrong-path case, not a hypothetical

- The final-answer text alone gives no signal anything is wrong -- it reads
  as a complete, well-grounded, competently formatted response.
- The system prompt's honest-gap instruction ("if the retrieved policy
  context says nothing is relevant, tell the customer plainly") only fires
  when retrieval returns *nothing*. Here retrieval returned *something*
  (the suspension doc), so the honest-gap guard never triggered, even though
  the something it returned didn't actually cover the cancellation-mechanics
  half of the question.
- Only the trajectory -- specifically, `retrieve_policy`'s
  `retrieved_doc_ids` output -- exposes the gap: what was actually
  retrieved vs. what the final answer claims to be based on.

## Root cause

A single-topic retrieval query facing a two-topic customer message. With
`top_k=2` and a fused-score threshold, the retriever needs *both* relevant
docs to independently clear the bar; here only one did. This is a real
limitation of top-k retrieval on compound queries, not a bug in the fusion
scoring itself.

## How to reproduce / inspect in Langfuse

1. cloud.langfuse.com -> `ecommerce-support-agent` project -> **Tracing**.
2. Open trace `119d9e059b6127e0e81673297fab3327`.
3. Click the `retrieve_policy` (RETRIEVER) span -> Output ->
   `retrieved_doc_ids: ['account_suspension_appeal']`.
4. Click the `handle_turn` (AGENT) span or the final `decide` (GENERATION)
   span -> Output -> the full final answer text with the fabricated
   cancellation steps.
5. Compare the two outputs side by side: the claimed cancellation mechanics
   don't trace back to anything in the retrieval output.

To send this exact query again against the deployed agent:

```bash
curl -X POST http://<ALB_DNS>/chat \
  -H "Content-Type: application/json" \
  -d '{"customer_id": "CUST005", "message": "My account is under review and I want to stop being charged every month, what do I do?"}'
```
