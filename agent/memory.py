"""Two kinds of memory: short-term (this conversation) and long-term
(this customer's history across past tickets, independent of this
conversation).
"""
import json
from pathlib import Path


class ShortTermMemory:
    """The in-conversation message buffer, in OpenAI chat-message format."""

    def __init__(self):
        self.messages: list[dict] = []

    def add(self, message: dict):
        self.messages.append(message)

    def as_list(self) -> list[dict]:
        return self.messages


class LongTermMemory:
    """Lookup against a mock prior-ticket-history store, keyed by customer_id."""

    def __init__(self, path: str | Path):
        self._data = json.loads(Path(path).read_text(encoding="utf-8"))

    def get_history_summary(self, customer_id: str) -> str:
        tickets = self._data.get(customer_id, [])
        if not tickets:
            return "No prior support tickets on file for this customer."
        lines = [f"- [{t['date']}] ({t['type']}) {t['summary']}" for t in tickets]
        return "Prior ticket history for this customer:\n" + "\n".join(lines)
