"""Cost report: queries agent/cost_ledger.py's SQLite ledger and reports a
real spend/token figure across whatever tickets have actually been run
through the agent (typically: the full eval suite), plus which ticket type
or code path is the most expensive one -- a real number, not an estimate
of an estimate.

Cost figures themselves ARE an estimate (see cost_ledger.py's per-model
price table docstring) -- Groq/Bedrock don't return a billed dollar amount
per call. This script reports what the ledger has recorded, honestly
labeled.

Usage:
    python -m eval.cost_report
"""
from __future__ import annotations

from agent.cost_ledger import CostLedger
from eval.fixtures import TICKETS

_TICKET_TYPE_BY_ID = {t["id"]: t["ticket_type"] for t in TICKETS}


def main():
    ledger = CostLedger()
    total_spend = ledger.total_spend()
    total_tokens = ledger.total_tokens()
    by_step = ledger.spend_by_step()
    by_ticket = ledger.spend_by_ticket()

    print("=" * 70)
    print("COST REPORT (estimated from published per-token rates, not billed truth)")
    print("=" * 70)
    print(f"Total spend:  ${total_spend:.6f}")
    print(f"Total tokens: {total_tokens}")
    print()
    print("Spend by step:")
    for step, cost in by_step.items():
        print(f"  {step:<20} ${cost:.6f}")
    print()
    print("Spend by ticket:")
    by_type: dict[str, float] = {}
    for ticket_id, cost in sorted(by_ticket.items(), key=lambda kv: -kv[1]):
        ticket_type = _TICKET_TYPE_BY_ID.get(ticket_id, "unknown")
        by_type[ticket_type] = by_type.get(ticket_type, 0.0) + cost
        print(f"  {ticket_id:<32} {ticket_type:<20} ${cost:.6f}")
    print()
    print("Spend by ticket TYPE:")
    most_expensive_type = None
    most_expensive_cost = -1.0
    for ticket_type, cost in sorted(by_type.items(), key=lambda kv: -kv[1]):
        print(f"  {ticket_type:<24} ${cost:.6f}")
        if cost > most_expensive_cost:
            most_expensive_type, most_expensive_cost = ticket_type, cost
    if most_expensive_type:
        print()
        print(f"Most expensive ticket type: {most_expensive_type} (${most_expensive_cost:.6f})")
    print("=" * 70)


if __name__ == "__main__":
    main()
