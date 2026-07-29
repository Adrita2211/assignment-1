"""Ticket-type classification.

Deliberately a deterministic keyword heuristic rather than an LLM call: the
four categories are lexically distinct enough in practice, and a
deterministic classifier is auditable (same input -> same category, every
time) and free of an extra API round trip. See README "why I built the
harness this way" for the fuller justification. Falls back to "general" when
no keyword matches, rather than forcing a guess into one of the four.
"""

CATEGORY_KEYWORDS = {
    "refund_request": [
        "refund", "money back", "reimburse", "return my money", "charge back",
    ],
    "delivery_issue": [
        "late", "delayed", "delay", "lost", "missing", "never arrived",
        "damaged", "broken", "hasn't arrived", "hasn't shown up",
    ],
    "order_status": [
        "where is my order", "order status", "track", "tracking",
        "when will", "shipped yet", "still processing",
    ],
    "subscription_account": [
        "subscription", "cancel my", "my account", "suspended", "flagged",
        "appeal", "password", "plus plan", "billing period",
    ],
}

VALID_CATEGORIES = tuple(CATEGORY_KEYWORDS) + ("general",)


def classify_ticket(text: str) -> str:
    text_l = text.lower()
    scores = {
        cat: sum(1 for kw in kws if kw in text_l)
        for cat, kws in CATEGORY_KEYWORDS.items()
    }
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return "general"
    return best
