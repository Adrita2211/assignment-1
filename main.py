"""Interactive CLI entrypoint.

Simulates an authenticated support session: you pick a customer_id up front
(standing in for whatever login/session system a real storefront would have
already resolved), then chat turn by turn. Ctrl+C or "exit" to quit.
"""
import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from agent.harness import SupportHarness

DATA_DIR = Path(__file__).resolve().parent / "data"


def _load_accounts() -> dict:
    return json.loads((DATA_DIR / "accounts.json").read_text(encoding="utf-8"))


async def main():
    load_dotenv()

    if not os.environ.get("GROQ_API_KEY"):
        print(
            "GROQ_API_KEY is not set. Copy .env.example to .env and put your "
            "Groq API key in it (https://console.groq.com/keys), then run this again."
        )
        sys.exit(1)

    accounts = _load_accounts()
    print("E-commerce Support Agent (Groq-backed)")
    print(f"Known customers: {', '.join(sorted(accounts))}")
    customer_id = input("Log in as customer_id: ").strip().upper()
    if customer_id not in accounts:
        print(f"Unknown customer_id {customer_id!r}. Exiting.")
        sys.exit(1)

    standing = accounts[customer_id]["standing"]
    print(f"\nLogged in as {customer_id}. Account standing: {standing.upper()}.")
    if standing == "flagged":
        print(
            "  Note: this account is FLAGGED pending review. Orders can still be "
            "placed; new activity may be scrutinized."
        )
    elif standing == "suspended":
        print(
            "  Note: this account is SUSPENDED. New orders cannot be placed. The "
            "customer can still view past orders, request refunds on past orders, "
            "and file a suspension appeal."
        )
    print("Type your message, or 'exit' to quit.\n")

    async with SupportHarness(customer_id) as harness:
        while True:
            try:
                user_text = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not user_text:
                continue
            if user_text.lower() in {"exit", "quit"}:
                break

            reply = await harness.handle_turn(user_text)
            print(f"agent> {reply}\n")


if __name__ == "__main__":
    asyncio.run(main())
