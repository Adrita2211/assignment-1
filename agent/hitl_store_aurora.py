"""Aurora-backed (RDS Data API) store for agent/hitl.py's PendingAction
state machine -- the real, deployable replacement for
agent/hitl_store.py's local SQLite version (Assignment 3 §2.5, "backed by
AgentCore's managed session state").

Why Aurora and not a new service: this project already provisions an
Aurora Serverless v2 cluster for the Bedrock Knowledge Base's pgvector
store (agent/rag_bedrock_kb.py) -- reusing that cluster for the HITL and
cost-ledger tables (agent/cost_ledger_aurora.py) means one piece of
managed-database infrastructure to provision and tear down, not two.

Why the RDS Data API and not a direct psycopg connection (contrast with
agent/rag_pgvector.py, which does connect directly): AgentCore Runtime
invocations are not guaranteed to run inside this cluster's VPC without
extra VPC networking configuration on the Runtime itself. The Data API is
plain HTTPS + IAM auth -- no VPC connectivity required from the caller --
which is exactly why the Knowledge Base's own storage configuration also
uses it (see this project's README).

Why a partial unique index instead of a DynamoDB-style lock item: Postgres
gives this for free and it is the idiomatic way to express "at most one
active pending action per resource" as a real, DB-enforced constraint
rather than application-level check-then-insert logic:
    CREATE UNIQUE INDEX ... ON hitl_pending_actions (resource_id)
    WHERE status = 'pending';
create_pending() below relies on this index and catches the resulting
unique-violation to raise DuplicatePendingActionError -- this is a
genuinely atomic version of the exact check that was missing in the
original found-and-fixed "Double Refund" bug (see eval/hitl_bug_repro.py),
not just a port of the SQLite check-then-insert pattern.

Same public interface as HITLStore (agent/hitl_store.py), so
agent/harness.py's calling code needs zero changes -- HITL_BACKEND=aurora
picks this class instead.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from agent.hitl import (
    ApprovalAlreadyDecidedError,
    ApprovalExpiredError,
    ApprovalStatus,
    DuplicatePendingActionError,
    PendingAction,
    StaleStateError,
)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS hitl_pending_actions (
    approval_id TEXT PRIMARY KEY,
    resource_id TEXT NOT NULL,
    ticket_id TEXT NOT NULL,
    customer_id TEXT NOT NULL,
    action TEXT NOT NULL,
    amount_usd NUMERIC NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    resource_snapshot JSONB NOT NULL,
    decided_at TEXT,
    decided_by TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS hitl_pending_actions_active_resource_idx
    ON hitl_pending_actions (resource_id) WHERE status = 'pending';
"""


class HITLStoreAurora:
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
        for stmt in SCHEMA_SQL.strip().split(";\n"):
            stmt = stmt.strip().rstrip(";")
            if stmt:
                self._execute(stmt)
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

    def _row_to_action(self, row: list[dict]) -> PendingAction:
        v = [self._val(f) for f in row]
        return PendingAction(
            approval_id=v[0], resource_id=v[1], ticket_id=v[2], customer_id=v[3],
            action=v[4], amount_usd=float(v[5]), status=ApprovalStatus(v[6]),
            created_at=v[7], expires_at=v[8], resource_snapshot=json.loads(v[9]),
            decided_at=v[10], decided_by=v[11],
        )

    _COLUMNS = ("approval_id, resource_id, ticket_id, customer_id, action, amount_usd, "
                "status, created_at, expires_at, resource_snapshot, decided_at, decided_by")

    def create_pending(self, action: PendingAction) -> PendingAction:
        from botocore.exceptions import ClientError

        self._ensure_schema()
        try:
            self._execute(
                f"INSERT INTO hitl_pending_actions ({self._COLUMNS}) VALUES ("
                ":approval_id, :resource_id, :ticket_id, :customer_id, :action, :amount_usd, "
                ":status, :created_at, :expires_at, :resource_snapshot::jsonb, :decided_at, :decided_by)",
                [
                    {"name": "approval_id", "value": {"stringValue": action.approval_id}},
                    {"name": "resource_id", "value": {"stringValue": action.resource_id}},
                    {"name": "ticket_id", "value": {"stringValue": action.ticket_id}},
                    {"name": "customer_id", "value": {"stringValue": action.customer_id}},
                    {"name": "action", "value": {"stringValue": action.action}},
                    {"name": "amount_usd", "value": {"doubleValue": action.amount_usd}},
                    {"name": "status", "value": {"stringValue": action.status.value}},
                    {"name": "created_at", "value": {"stringValue": action.created_at}},
                    {"name": "expires_at", "value": {"stringValue": action.expires_at}},
                    {"name": "resource_snapshot", "value": {"stringValue": json.dumps(action.resource_snapshot)}},
                    {"name": "decided_at", "value": {"isNull": True} if action.decided_at is None else {"stringValue": action.decided_at}},
                    {"name": "decided_by", "value": {"isNull": True} if action.decided_by is None else {"stringValue": action.decided_by}},
                ],
            )
        except ClientError as e:
            msg = str(e)
            if "duplicate key value violates unique constraint" in msg and "active_resource_idx" in msg:
                existing = self.get_active_for_resource(action.resource_id)
                raise DuplicatePendingActionError(
                    f"resource_id={action.resource_id!r} already has an active pending "
                    f"action (approval_id={existing.approval_id if existing else '?'!r}, "
                    f"requested for ticket_id={existing.ticket_id if existing else '?'!r}) "
                    "-- refusing to open a second one"
                )
            raise
        return action

    def get(self, approval_id: str) -> PendingAction | None:
        self._ensure_schema()
        resp = self._execute(
            f"SELECT {self._COLUMNS} FROM hitl_pending_actions WHERE approval_id = :approval_id",
            [{"name": "approval_id", "value": {"stringValue": approval_id}}],
        )
        records = resp.get("records", [])
        return self._row_to_action(records[0]) if records else None

    def get_active_for_resource(self, resource_id: str) -> PendingAction | None:
        self._ensure_schema()
        resp = self._execute(
            f"SELECT {self._COLUMNS} FROM hitl_pending_actions "
            "WHERE resource_id = :resource_id AND status = 'pending' "
            "ORDER BY created_at DESC LIMIT 1",
            [{"name": "resource_id", "value": {"stringValue": resource_id}}],
        )
        records = resp.get("records", [])
        return self._row_to_action(records[0]) if records else None

    def decide(self, approval_id: str, status: ApprovalStatus, decided_by: str) -> PendingAction:
        action = self.get(approval_id)
        if action is None:
            raise ValueError(f"No pending action with approval_id={approval_id!r}")
        if action.status != ApprovalStatus.PENDING:
            raise ApprovalAlreadyDecidedError(
                f"approval_id={approval_id!r} is already {action.status.value!r}, refusing to re-decide"
            )
        if datetime.now(timezone.utc) > datetime.fromisoformat(action.expires_at):
            self._execute(
                "UPDATE hitl_pending_actions SET status = 'expired' WHERE approval_id = :approval_id",
                [{"name": "approval_id", "value": {"stringValue": approval_id}}],
            )
            raise ApprovalExpiredError(f"approval_id={approval_id!r} expired at {action.expires_at}")

        decided_at = datetime.now(timezone.utc).isoformat()
        self._execute(
            "UPDATE hitl_pending_actions SET status = :status, decided_at = :decided_at, "
            "decided_by = :decided_by WHERE approval_id = :approval_id",
            [
                {"name": "status", "value": {"stringValue": status.value}},
                {"name": "decided_at", "value": {"stringValue": decided_at}},
                {"name": "decided_by", "value": {"stringValue": decided_by}},
                {"name": "approval_id", "value": {"stringValue": approval_id}},
            ],
        )
        return self.get(approval_id)

    def revalidate_before_execution(self, action: PendingAction, live_order: dict) -> None:
        snap = action.resource_snapshot
        if live_order["status"] != snap["status"] or live_order["order_total"] != snap["order_total"]:
            raise StaleStateError(
                f"order {action.resource_id} changed since approval was requested "
                f"(was status={snap['status']!r} total={snap['order_total']}, "
                f"now status={live_order['status']!r} total={live_order['order_total']}) "
                "-- refusing to execute against stale state, re-review required"
            )

    def mark_executed(self, approval_id: str) -> PendingAction:
        action = self.get(approval_id)
        if action is None:
            raise ValueError(f"No pending action with approval_id={approval_id!r}")
        if action.status != ApprovalStatus.APPROVED:
            raise ApprovalAlreadyDecidedError(
                f"approval_id={approval_id!r} is {action.status.value!r}, not APPROVED -- refusing to execute"
            )
        self._execute(
            "UPDATE hitl_pending_actions SET status = 'executed' WHERE approval_id = :approval_id",
            [{"name": "approval_id", "value": {"stringValue": approval_id}}],
        )
        return self.get(approval_id)

    def list_pending(self) -> list[PendingAction]:
        self._ensure_schema()
        resp = self._execute(
            f"SELECT {self._COLUMNS} FROM hitl_pending_actions WHERE status = 'pending' ORDER BY created_at"
        )
        return [self._row_to_action(r) for r in resp.get("records", [])]

    def close(self) -> None:
        pass
