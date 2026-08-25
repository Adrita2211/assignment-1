"""Hand-rolled SQLite cost/token ledger -- deliberately not migrated to a
managed AWS service (see README's cost-tracking section): no managed
service reports per-ticket LLM token cost the way a purpose-built local
ledger can, so there's nothing worth outsourcing here, unlike retrieval
(Bedrock Knowledge Bases) or the policy boundary (Verified Permissions).

Logs one row per LLM API call (`decide_node` is the single choke point
every call passes through -- see agent/harness.py's own docstring on this),
tagged by ticket_id and step, so eval/cost_report.py can answer "which
ticket type or code path is actually the expensive one" with a real
queried number instead of an estimate.

Cost is *estimated* from a hard-coded per-model price table, not a billed
truth -- Groq/Bedrock don't return a dollar figure in their responses.
Documented as an estimate everywhere it's reported, not silently presented
as exact.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "cost_ledger.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id TEXT NOT NULL,
    step TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cost_usd REAL NOT NULL,
    latency_s REAL NOT NULL,
    error TEXT,
    created_at TEXT NOT NULL
);
"""

# $ per 1K tokens. Published/listed rates at time of writing, not queried
# live -- update here if pricing changes rather than computing per-call.
_PRICE_PER_1K: dict[str, tuple[float, float]] = {
    # (input, output)
    "openai/gpt-oss-120b": (0.05, 0.08),          # Groq-hosted, per Groq's published rate card
    "anthropic.claude-3-5-sonnet-20241022-v2:0": (0.003, 0.015),   # Bedrock Claude 3.5 Sonnet
}
_DEFAULT_PRICE_PER_1K = (0.0, 0.0)  # unknown model -- report $0 rather than guess


def _estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    in_rate, out_rate = _PRICE_PER_1K.get(model, _DEFAULT_PRICE_PER_1K)
    return round((input_tokens / 1000) * in_rate + (output_tokens / 1000) * out_rate, 6)


class CostLedger:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.db_path)
            self._conn.execute(SCHEMA)
            self._conn.commit()
        except (sqlite3.OperationalError, OSError) as exc:
            # Same reasoning as record()'s try/except -- must not crash
            # SupportHarness.__init__ (which every single turn depends on)
            # over a pure-telemetry write failure.
            print(f"[COST_LEDGER] init failed, cost tracking disabled for this session: {exc}")
            self._conn = None

    def record(
        self,
        *,
        ticket_id: str,
        step: str,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_s: float,
        error: str | None = None,
    ) -> None:
        if self._conn is None:
            return
        cost_usd = _estimate_cost_usd(model, input_tokens, output_tokens)
        try:
            self._conn.execute(
                """
                INSERT INTO llm_calls
                    (ticket_id, step, provider, model, input_tokens, output_tokens, cost_usd, latency_s, error, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ticket_id, step, provider, model, input_tokens, output_tokens,
                    cost_usd, latency_s, error, datetime.now(timezone.utc).isoformat(),
                ),
            )
            self._conn.commit()
        except sqlite3.OperationalError as exc:
            # Cost tracking is pure telemetry -- a failed write here (e.g.
            # the deployed container's filesystem being read-only for this
            # non-root process, a real problem this project hit once local
            # SQLite ran on AgentCore Runtime with no Aurora backend
            # configured) must never take down the customer-facing turn
            # itself. Contrast with agent/hitl_store.py, which does NOT
            # swallow write failures -- a lost approval record is a safety
            # issue, not a reporting gap, so that one stays fatal on purpose.
            print(f"[COST_LEDGER] record() failed, continuing without it: {exc}")

    def total_spend(self) -> float:
        row = self._conn.execute("SELECT COALESCE(SUM(cost_usd), 0) FROM llm_calls").fetchone()
        return round(row[0], 6)

    def total_tokens(self) -> int:
        row = self._conn.execute(
            "SELECT COALESCE(SUM(input_tokens + output_tokens), 0) FROM llm_calls"
        ).fetchone()
        return int(row[0])

    def spend_by_step(self) -> dict[str, float]:
        rows = self._conn.execute(
            "SELECT step, SUM(cost_usd) FROM llm_calls GROUP BY step ORDER BY SUM(cost_usd) DESC"
        ).fetchall()
        return {step: round(cost, 6) for step, cost in rows}

    def spend_by_ticket(self) -> dict[str, float]:
        rows = self._conn.execute(
            "SELECT ticket_id, SUM(cost_usd) FROM llm_calls GROUP BY ticket_id ORDER BY SUM(cost_usd) DESC"
        ).fetchall()
        return {ticket_id: round(cost, 6) for ticket_id, cost in rows}

    def close(self) -> None:
        self._conn.close()
