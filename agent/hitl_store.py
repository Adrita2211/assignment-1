"""Local SQLite-backed store for agent/hitl.py's PendingAction state
machine -- the pre-AWS, locally-testable stand-in for AgentCore's managed
session state (see README's HITL section for what AgentCore's managed
memory does and doesn't absorb). Same interface either way, so swapping
the backend later is a config change to this module, not a redesign of
the calling code in agent/harness.py.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from agent.hitl import (
    ApprovalAlreadyDecidedError,
    ApprovalExpiredError,
    ApprovalStatus,
    DuplicatePendingActionError,
    PendingAction,
    StaleStateError,
)

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "hitl_store.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS pending_actions (
    approval_id TEXT PRIMARY KEY,
    resource_id TEXT NOT NULL,
    ticket_id TEXT NOT NULL,
    customer_id TEXT NOT NULL,
    action TEXT NOT NULL,
    amount_usd REAL NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    resource_snapshot TEXT NOT NULL,
    decided_at TEXT,
    decided_by TEXT
);
"""


class HITLStore:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute(SCHEMA)
        self._conn.commit()

    def _row_to_action(self, row) -> PendingAction:
        return PendingAction(
            approval_id=row[0], resource_id=row[1], ticket_id=row[2], customer_id=row[3],
            action=row[4], amount_usd=row[5], status=ApprovalStatus(row[6]),
            created_at=row[7], expires_at=row[8], resource_snapshot=json.loads(row[9]),
            decided_at=row[10], decided_by=row[11],
        )

    def create_pending(self, action: PendingAction) -> PendingAction:
        # THE FIX for the real "Double Refund" bug (see eval/hitl_bug_repro.py
        # and the README's HITL section): v1 of this method had no check
        # here at all, so two overlapping tickets against the same order
        # could each get their own PendingAction, both get approved
        # independently, and both execute. Keying the uniqueness check off
        # resource_id (the order), not ticket_id or approval_id, is what
        # actually closes it -- two tickets about the same order must
        # collide here, even though they're otherwise unrelated requests.
        existing = self.get_active_for_resource(action.resource_id)
        if existing is not None:
            raise DuplicatePendingActionError(
                f"resource_id={action.resource_id!r} already has an active pending "
                f"action (approval_id={existing.approval_id!r}, requested for "
                f"ticket_id={existing.ticket_id!r}) -- refusing to open a second one"
            )
        self._conn.execute(
            """
            INSERT INTO pending_actions
                (approval_id, resource_id, ticket_id, customer_id, action, amount_usd,
                 status, created_at, expires_at, resource_snapshot, decided_at, decided_by)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                action.approval_id, action.resource_id, action.ticket_id, action.customer_id,
                action.action, action.amount_usd, action.status.value, action.created_at,
                action.expires_at, json.dumps(action.resource_snapshot), action.decided_at, action.decided_by,
            ),
        )
        self._conn.commit()
        return action

    def get(self, approval_id: str) -> PendingAction | None:
        row = self._conn.execute(
            "SELECT * FROM pending_actions WHERE approval_id = ?", (approval_id,)
        ).fetchone()
        return self._row_to_action(row) if row else None

    def get_active_for_resource(self, resource_id: str) -> PendingAction | None:
        row = self._conn.execute(
            "SELECT * FROM pending_actions WHERE resource_id = ? AND status = ? ORDER BY created_at DESC LIMIT 1",
            (resource_id, ApprovalStatus.PENDING.value),
        ).fetchone()
        return self._row_to_action(row) if row else None

    def decide(self, approval_id: str, status: ApprovalStatus, decided_by: str) -> PendingAction:
        action = self.get(approval_id)
        if action is None:
            raise ValueError(f"No pending action with approval_id={approval_id!r}")
        if action.status != ApprovalStatus.PENDING:
            raise ApprovalAlreadyDecidedError(
                f"approval_id={approval_id!r} is already {action.status.value!r}, refusing to re-decide"
            )
        if datetime.now(timezone.utc) > datetime.fromisoformat(action.expires_at):
            self._conn.execute(
                "UPDATE pending_actions SET status = ? WHERE approval_id = ?",
                (ApprovalStatus.EXPIRED.value, approval_id),
            )
            self._conn.commit()
            raise ApprovalExpiredError(f"approval_id={approval_id!r} expired at {action.expires_at}")

        decided_at = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "UPDATE pending_actions SET status = ?, decided_at = ?, decided_by = ? WHERE approval_id = ?",
            (status.value, decided_at, decided_by, approval_id),
        )
        self._conn.commit()
        return self.get(approval_id)

    def revalidate_before_execution(self, action: PendingAction, live_order: dict) -> None:
        """The single most common real HITL bug: a gate that approves
        against a frozen snapshot and then executes without re-checking
        current state can approve against state that's since changed.
        Compares the fields that actually matter for a refund decision --
        status and order_total -- against what was true when the approval
        was requested, refusing to proceed silently if either differs."""
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
        self._conn.execute(
            "UPDATE pending_actions SET status = ? WHERE approval_id = ?",
            (ApprovalStatus.EXECUTED.value, approval_id),
        )
        self._conn.commit()
        return self.get(approval_id)

    def list_pending(self) -> list[PendingAction]:
        rows = self._conn.execute(
            "SELECT * FROM pending_actions WHERE status = ? ORDER BY created_at", (ApprovalStatus.PENDING.value,)
        ).fetchall()
        return [self._row_to_action(r) for r in rows]

    def close(self) -> None:
        self._conn.close()
