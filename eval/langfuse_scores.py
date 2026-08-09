"""Pushes each eval script's already-computed verdict to Langfuse as a Score
attached to that ticket's trace (SupportHarness.last_trace_id, threaded
through by eval/_run_utils.py's run_ticket). The eval scripts already derive
every verdict below -- see each script's own docstring for the taxonomy axis
it covers; this module's only job is reporting them to Langfuse's Scores
tab, not re-deriving them.

Score naming mirrors the "who's scoring / what's scored" taxonomy so the
Scores tab is self-explanatory without cross-referencing this file:

  name                       who        what          ref-based?
  -------------------------  ---------  ------------  ----------
  trajectory_pass            rule       trajectory    yes (required_tools)
  policy_adherence           rule       trajectory    yes (audit_log)
  safety                     rule       final-answer  yes (known leak strings)
  calibration_retrieval      rule       final-answer  yes (retrieval threshold)
  calibration_judge          llm-judge  final-answer  free-form
  calibration_judge_agreement llm-judge (self-consistency across N runs)
  robustness_pass            rule       trajectory    yes (baseline ticket)
  efficiency_latency_s /
  efficiency_total_tokens    rule       efficiency    yes (harness.metrics())

Human Eval (taxonomy: who's-scoring #3) is deliberately not wired here --
it's an inherently manual annotation step done in the Langfuse UI's Human
Annotation tab against these same traces, not something a script produces.

create_score() already no-ops internally when tracing is disabled (no
LANGFUSE_PUBLIC_KEY), so push_score() below stays a plain passthrough --
eval scripts keep working standalone; scoring is strictly additive.
"""
from __future__ import annotations

from typing import Optional, Union

from agent.tracing import langfuse


def push_score(
    trace_id: Optional[str],
    name: str,
    value: Union[float, int, bool, str],
    data_type: str = "BOOLEAN",
    comment: Optional[str] = None,
) -> None:
    """Attach one score to one trace. A missing trace_id (e.g. tracing was
    disabled for this run) is silently skipped rather than raised -- a score
    with nothing to attach to isn't an error, it's just not applicable."""
    if not trace_id:
        return
    langfuse.create_score(
        trace_id=trace_id,
        name=name,
        value=value,
        data_type=data_type,
        comment=comment,
    )
