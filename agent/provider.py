"""LLM provider abstraction.

Isolates the model SDK behind one method: call(messages, tools) ->
normalized {"text", "tool_calls"} dict. Nothing else in the codebase imports
`groq` directly. agent/harness.py and everything downstream only ever deals
with that normalized shape, regardless of which concrete provider is behind
it -- the seam this assignment's optional §2.6.4 Bedrock-readiness bonus
asks for (see BedrockProvider below).

Groq's chat-completions API is OpenAI-compatible, so `messages` and `tools`
already use the OpenAI wire format (role-based messages, tools as
{"type": "function", "function": {...}}).
"""
import json
import os
import re
import time
from abc import ABC, abstractmethod

from groq import Groq, RateLimitError

DEFAULT_MODEL = "openai/gpt-oss-120b"
MAX_RATE_LIMIT_RETRIES = 5

_RETRY_MS_RE = re.compile(r"try again in (\d+(?:\.\d+)?)(ms|s)", re.IGNORECASE)


def _rate_limit_wait_s(exc: RateLimitError, attempt: int) -> float:
    """Groq's 429 body names its own suggested wait ("Please try again in
    127.5ms") -- honor that when present instead of a blind fixed backoff,
    since the free tier's per-minute window means the real wait is often
    sub-second, not the many-seconds an exponential backoff would default to.
    Falls back to attempt-indexed exponential backoff if the message shape
    ever changes."""
    message = str(exc)
    match = _RETRY_MS_RE.search(message)
    if match:
        value, unit = match.groups()
        seconds = float(value) / 1000 if unit.lower() == "ms" else float(value)
        return seconds + 0.1  # small margin so the retry lands just after the window resets
    return min(2 ** attempt, 30)


class LLMProvider(ABC):
    """The seam every concrete provider implements. Kept intentionally
    small -- one method, the OpenAI-compatible wire format in, the
    normalized {"text", "tool_calls"} shape out -- so agent/harness.py never
    needs to know which provider it's holding."""

    model: str

    @abstractmethod
    def call(self, messages: list[dict], tools: list[dict] | None = None, max_tokens: int = 1024) -> dict:
        """Returns {"text", "tool_calls", "usage": {"input_tokens", "output_tokens"}}.
        `usage` is the System-observability-layer signal (see agent/harness.py's
        SupportHarness.metrics()) -- cost tracking needs real token counts, not
        an estimate, so every concrete provider fills it from the SDK response
        rather than leaving it to the caller to guess."""
        raise NotImplementedError


class GroqProvider(LLMProvider):
    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None):
        self.client = Groq(api_key=api_key or os.environ["GROQ_API_KEY"])
        self.model = model

    def call(self, messages: list[dict], tools: list[dict] | None = None, max_tokens: int = 1024) -> dict:
        kwargs = dict(model=self.model, messages=messages, max_tokens=max_tokens)
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = self._create_with_retry(kwargs)
        msg = response.choices[0].message

        tool_calls = []
        for tc in (msg.tool_calls or []):
            try:
                arguments = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, TypeError):
                arguments = {}
            tool_calls.append({"id": tc.id, "name": tc.function.name, "arguments": arguments})

        usage = getattr(response, "usage", None)
        return {
            "text": msg.content,
            "tool_calls": tool_calls,
            "usage": {
                "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                "output_tokens": getattr(usage, "completion_tokens", 0) or 0,
            },
        }

    def _create_with_retry(self, kwargs: dict):
        """The free tier's 8000 TPM cap is easy to cross running a 10+
        ticket eval suite (or several tickets back to back in CI)
        sequentially -- retrying the specific 429 case, honoring the API's
        own suggested wait, keeps that from surfacing as a flaky eval/CI
        failure that has nothing to do with the agent itself."""
        for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
            try:
                return self.client.chat.completions.create(**kwargs)
            except RateLimitError as exc:
                if attempt == MAX_RATE_LIMIT_RETRIES:
                    raise
                time.sleep(_rate_limit_wait_s(exc, attempt))


class BedrockProvider(LLMProvider):
    """Not implemented -- a code-shape exercise per the assignment's
    optional §2.6.4, not a deployed endpoint. Documents exactly which call
    would go here and which env var would select it, so the seam exists and
    is reviewable without pretending it's functional.

    Selected via LLM_PROVIDER=bedrock (see get_provider() below). When
    implemented, `call()` would invoke the Bedrock Runtime `converse` API
    (boto3 `bedrock-runtime` client's `converse()`, or the lower-level
    `invoke_model()` for a model without Converse API support), translating
    this project's OpenAI-style `messages`/`tools` into the Converse API's
    `messages`/`toolConfig` shapes and translating the response back into
    this class's {"text", "tool_calls"} normalized form -- the same
    translation GroqProvider.call() already does for Groq's wire format.
    """

    def __init__(self, model: str | None = None):
        self.model = model or os.environ.get("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20241022-v2:0")

    def call(self, messages: list[dict], tools: list[dict] | None = None, max_tokens: int = 1024) -> dict:
        raise NotImplementedError(
            "BedrockProvider is a documented seam, not a working provider yet -- see this class's "
            "docstring for the exact bedrock-runtime call it would make. When implemented, `usage` "
            "would come from the converse() response's `usage.inputTokens` / `usage.outputTokens`."
        )


def get_provider(name: str | None = None) -> LLMProvider:
    """LLM_PROVIDER env var selects the backend; defaults to groq, this
    project's only functional provider today."""
    name = name or os.environ.get("LLM_PROVIDER", "groq")
    if name == "groq":
        return GroqProvider()
    if name == "bedrock":
        return BedrockProvider()
    raise ValueError(f"Unknown provider: {name!r} (expected 'groq' or 'bedrock')")
