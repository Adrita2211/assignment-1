"""Aurora-backed (RDS Data API) cost/token ledger -- the real, deployable
replacement for agent/cost_ledger.py's local SQLite version, for the same
reason as agent/hitl_store_aurora.py: AgentCore Runtime invocations don't
share persistent local disk, so a SQLite file resets per invocation and
silently stops accumulating spend once this actually runs on AgentCore.
Reuses the same Aurora cluster the Bedrock Knowledge Base already
provisions (see agent/hitl_store_aurora.py's module docstring for why one
shared cluster instead of a second managed service).

Same public interface as CostLedger (record/total_spend/total_tokens/
spend_by_step/spend_by_ticket/close) -- COST_LEDGER_BACKEND=aurora picks
this class instead of the local SQLite one.
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS cost_ledger_calls (
    call_id TEXT PRIMARY KEY,
    ticket_id TEXT NOT NULL,
    step TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cost_usd NUMERIC NOT NULL,
    latency_s NUMERIC NOT NULL,
    error TEXT,
    created_at TEXT NOT NULL
);
"""

_PRICE_PER_1K: dict[str, tuple[float, float]] = {
    "openai/gpt-oss-120b": (0.05, 0.08),
    "anthropic.claude-3-5-sonnet-20241022-v2:0": (0.003, 0.015),
    "amazon.nova-lite-v1:0": (0.00006, 0.00024),
    "amazon.nova-micro-v1:0": (0.000035, 0.00014),
}
_DEFAULT_PRICE_PER_1K = (0.0, 0.0)


def _estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    in_rate, out_rate = _PRICE_PER_1K.get(model, _DEFAULT_PRICE_PER_1K)
    return round((input_tokens / 1000) * in_rate + (output_tokens / 1000) * out_rate, 6)


class CostLedgerAurora:
    def __init__(self, cluster_arn: str | None = None, secret_arn: str | None = None,
                 database: str | None = None, region: str | None = None):
        self.cluster_arn = cluster_arn or os.environ["AURORA_CLUSTER_ARN"]
        self.secret_arn = secret_arn or os.environ["AURORA_SECRET_ARN"]
        self.database = database or os.environ.get("AURORA_DATABASE", "kbdb")
        self._region = region or os.environ.get("AWS_REGION", "us-east-1")
        self._client = None
        self._schema_ready = False

    @property
    def client(self):
        if self._client is None:
            import boto3

            self._client = boto3.client("rds-data", region_name=self._region)
        return self._client

    def _ensure_schema(self) -> None:
        if self._schema_ready:
            return
        self._execute(SCHEMA_SQL.strip())
        self._schema_ready = True

    def _execute(self, sql: str, params: list[dict] | None = None) -> dict:
        kwargs = {
            "resourceArn": self.cluster_arn,
            "secretArn": self.secret_arn,
            "database": self.database,
            "sql": sql,
        }
        if params:
            kwargs["parameters"] = params
        return self.client.execute_statement(**kwargs)

    @staticmethod
    def _val(field: dict):
        if field.get("isNull"):
            return None
        for key in ("stringValue", "longValue", "doubleValue", "booleanValue"):
            if key in field:
                return field[key]
        return None

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
        self._ensure_schema()
        cost_usd = _estimate_cost_usd(model, input_tokens, output_tokens)
        self._execute(
            "INSERT INTO cost_ledger_calls "
            "(call_id, ticket_id, step, provider, model, input_tokens, output_tokens, "
            "cost_usd, latency_s, error, created_at) VALUES "
            "(:call_id, :ticket_id, :step, :provider, :model, :input_tokens, :output_tokens, "
            ":cost_usd, :latency_s, :error, :created_at)",
            [
                {"name": "call_id", "value": {"stringValue": str(uuid.uuid4())}},
                {"name": "ticket_id", "value": {"stringValue": ticket_id}},
                {"name": "step", "value": {"stringValue": step}},
                {"name": "provider", "value": {"stringValue": provider}},
                {"name": "model", "value": {"stringValue": model}},
                {"name": "input_tokens", "value": {"longValue": input_tokens}},
                {"name": "output_tokens", "value": {"longValue": output_tokens}},
                {"name": "cost_usd", "value": {"doubleValue": cost_usd}},
                {"name": "latency_s", "value": {"doubleValue": latency_s}},
                {"name": "error", "value": {"isNull": True} if error is None else {"stringValue": error}},
                {"name": "created_at", "value": {"stringValue": datetime.now(timezone.utc).isoformat()}},
            ],
        )

    def total_spend(self) -> float:
        self._ensure_schema()
        resp = self._execute("SELECT COALESCE(SUM(cost_usd), 0) FROM cost_ledger_calls")
        return round(float(self._val(resp["records"][0][0])), 6)

    def total_tokens(self) -> int:
        self._ensure_schema()
        resp = self._execute("SELECT COALESCE(SUM(input_tokens + output_tokens), 0) FROM cost_ledger_calls")
        return int(self._val(resp["records"][0][0]))

    def spend_by_step(self) -> dict[str, float]:
        self._ensure_schema()
        resp = self._execute(
            "SELECT step, SUM(cost_usd) FROM cost_ledger_calls GROUP BY step ORDER BY SUM(cost_usd) DESC"
        )
        return {self._val(r[0]): round(float(self._val(r[1])), 6) for r in resp.get("records", [])}

    def spend_by_ticket(self) -> dict[str, float]:
        self._ensure_schema()
        resp = self._execute(
            "SELECT ticket_id, SUM(cost_usd) FROM cost_ledger_calls GROUP BY ticket_id ORDER BY SUM(cost_usd) DESC"
        )
        return {self._val(r[0]): round(float(self._val(r[1])), 6) for r in resp.get("records", [])}

    def close(self) -> None:
        pass
