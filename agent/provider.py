"""LLM provider abstraction.

Isolates the Groq SDK behind one function: call(messages, tools) ->
normalized dict. Nothing else in the codebase imports `groq` directly. If
this project later swaps to Anthropic or OpenAI, only this file changes --
the harness and everything downstream just deals with the normalized
{"text", "tool_calls"} shape.

Groq's chat-completions API is OpenAI-compatible, so `messages` and `tools`
already use the OpenAI wire format (role-based messages, tools as
{"type": "function", "function": {...}}).
"""
import json
import os

from groq import Groq

DEFAULT_MODEL = "openai/gpt-oss-120b"


class GroqProvider:
    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None):
        self.client = Groq(api_key=api_key or os.environ["GROQ_API_KEY"])
        self.model = model

    def call(self, messages: list[dict], tools: list[dict] | None = None, max_tokens: int = 1024) -> dict:
        kwargs = dict(model=self.model, messages=messages, max_tokens=max_tokens)
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = self.client.chat.completions.create(**kwargs)
        msg = response.choices[0].message

        tool_calls = []
        for tc in (msg.tool_calls or []):
            try:
                arguments = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, TypeError):
                arguments = {}
            tool_calls.append({"id": tc.id, "name": tc.function.name, "arguments": arguments})

        return {"text": msg.content, "tool_calls": tool_calls}
