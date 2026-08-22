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


def _to_bedrock_messages(messages: list[dict]) -> tuple[str | None, list[dict]]:
    """OpenAI-shaped messages -> Converse API's messages shape. Converse
    takes the system prompt SEPARATELY (a `system` param, not a message
    with role="system"), and every message's content is a list of typed
    blocks ({"text": ...} / {"toolUse": ...} / {"toolResult": ...}), not a
    bare string. Returns (system_text, converse_messages)."""
    system_text = None
    converse_messages = []
    for m in messages:
        if m["role"] == "system":
            system_text = m["content"]
            continue
        if m["role"] == "tool":
            # A tool RESULT -- Converse expects this as a "user" turn
            # carrying a toolResult block, not its own role.
            converse_messages.append({
                "role": "user",
                "content": [{
                    "toolResult": {
                        "toolUseId": m["tool_call_id"],
                        "content": [{"text": m["content"]}],
                    },
                }],
            })
        elif m["role"] == "assistant" and m.get("tool_calls"):
            content = []
            if m.get("content"):
                content.append({"text": m["content"]})
            for tc in m["tool_calls"]:
                content.append({
                    "toolUse": {
                        "toolUseId": tc["id"],
                        "name": tc["function"]["name"],
                        "input": json.loads(tc["function"]["arguments"]),
                    },
                })
            converse_messages.append({"role": "assistant", "content": content})
        else:
            converse_messages.append({"role": m["role"], "content": [{"text": m["content"] or ""}]})
    return system_text, converse_messages


def _to_bedrock_tool_config(tools: list[dict]) -> dict:
    """OpenAI-shaped tools ({"type": "function", "function": {...}}) ->
    Converse's toolConfig shape ({"tools": [{"toolSpec": {...}}]}). The
    JSON-schema `parameters` block is identical either way -- this is
    genuinely just a wrapper-shape translation, not a schema rewrite,
    which is exactly why agent/harness.py's TOOLS didn't need to change."""
    return {
        "tools": [
            {
                "toolSpec": {
                    "name": t["function"]["name"],
                    "description": t["function"]["description"],
                    "inputSchema": {"json": t["function"]["parameters"]},
                },
            }
            for t in tools
        ],
    }


def _from_bedrock_response(response: dict) -> dict:
    """Converse API response -> this project's normalized {"text",
    "tool_calls", "usage"} shape, the same contract GroqProvider.call()
    already returns."""
    message = response["output"]["message"]
    text_parts = []
    tool_calls = []
    for block in message.get("content", []):
        if "text" in block:
            text_parts.append(block["text"])
        elif "toolUse" in block:
            tu = block["toolUse"]
            tool_calls.append({"id": tu["toolUseId"], "name": tu["name"], "arguments": tu["input"]})

    usage = response.get("usage", {})
    return {
        "text": "\n".join(text_parts) if text_parts else None,
        "tool_calls": tool_calls,
        "usage": {
            "input_tokens": usage.get("inputTokens", 0),
            "output_tokens": usage.get("outputTokens", 0),
        },
    }


class BedrockProvider(LLMProvider):
    """Bedrock Runtime `converse()` API -- the AWS-managed path this
    project's agent runs on once wrapped in agentcore_app.py's
    BedrockAgentCoreApp (Assignment 3 §2.2). Selected via LLM_PROVIDER=bedrock
    (see get_provider() below).

    Local dev keeps using GroqProvider (LLM_PROVIDER unset/groq); this
    class only activates with real AWS credentials configured, and needs
    the target model actually enabled for this account in the Bedrock
    console -- model access has real approval lead time for some models
    (see this project's README for the exact model ID this deployment
    uses and when it was confirmed enabled, since Bedrock model
    availability changes per-account and isn't something to hardcode a
    permanent default for here).
    """

    def __init__(self, model: str | None = None, region: str | None = None):
        self.model = model or os.environ.get(
            "BEDROCK_MODEL_ID",
            # A known-good, generally-available model ID as a last-resort
            # fallback if BEDROCK_MODEL_ID isn't set -- NOT necessarily the
            # model this deployment actually uses. Set BEDROCK_MODEL_ID
            # explicitly to whatever this account's confirmed-enabled model
            # ID is (check `aws bedrock list-foundation-models` or the
            # console) rather than relying on this default.
            "anthropic.claude-3-5-sonnet-20241022-v2:0",
        )
        self._region = region or os.environ.get("AWS_REGION", "us-east-1")
        self._client = None

    @property
    def client(self):
        # Lazy: importing boto3 and creating a client at construction time
        # would make every BedrockProvider() instantiation (including ones
        # created just to read .model, e.g. in tests) require real AWS
        # credentials/network access even when call() is never invoked.
        if self._client is None:
            import boto3

            self._client = boto3.client("bedrock-runtime", region_name=self._region)
        return self._client

    def call(self, messages: list[dict], tools: list[dict] | None = None, max_tokens: int = 1024) -> dict:
        system_text, converse_messages = _to_bedrock_messages(messages)
        kwargs = dict(
            modelId=self.model,
            messages=converse_messages,
            inferenceConfig={"maxTokens": max_tokens},
        )
        if system_text:
            kwargs["system"] = [{"text": system_text}]
        if tools:
            kwargs["toolConfig"] = _to_bedrock_tool_config(tools)

        response = self.client.converse(**kwargs)
        return _from_bedrock_response(response)


def get_provider(name: str | None = None) -> LLMProvider:
    """LLM_PROVIDER env var selects the backend; defaults to groq, this
    project's only functional provider today."""
    name = name or os.environ.get("LLM_PROVIDER", "groq")
    if name == "groq":
        return GroqProvider()
    if name == "bedrock":
        return BedrockProvider()
    raise ValueError(f"Unknown provider: {name!r} (expected 'groq' or 'bedrock')")
