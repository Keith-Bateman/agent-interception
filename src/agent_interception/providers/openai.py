"""OpenAI request/response/stream parser."""

from __future__ import annotations

import json
from typing import Any

from agent_interception.models import (
    CostEstimate,
    Provider,
    StreamChunk,
    TokenUsage,
)
from agent_interception.providers.base import ProviderParser

# Cost per million tokens (input, output) in USD
OPENAI_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-4": (30.00, 60.00),
    "gpt-3.5-turbo": (0.50, 1.50),
    "o1": (15.00, 60.00),
    "o1-mini": (3.00, 12.00),
    "o3-mini": (1.10, 4.40),
}


class OpenAIParser(ProviderParser):
    """Parser for OpenAI API format (also used by compatible providers)."""

    @property
    def provider(self) -> Provider:
        return Provider.OPENAI

    def parse_request(self, body: dict[str, Any]) -> dict[str, Any]:
        """Parse an OpenAI chat completion request."""
        messages = body.get("messages", [])

        system_prompt = None
        for msg in messages:
            if msg.get("role") == "system":
                content = msg.get("content", "")
                if isinstance(content, str):
                    system_prompt = content
                elif isinstance(content, list):
                    system_prompt = " ".join(
                        p.get("text", "") for p in content if isinstance(p, dict)
                    )
                break

        return {
            "model": body.get("model"),
            "system_prompt": system_prompt,
            "messages": messages,
            "tools": body.get("tools"),
            "is_streaming": body.get("stream", False),
            "image_metadata": self.extract_image_metadata(messages),
        }

    def parse_response(self, body: dict[str, Any]) -> dict[str, Any]:
        """Parse an OpenAI chat completion response."""
        result: dict[str, Any] = {"model": body.get("model")}

        choices = body.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            result["response_text"] = message.get("content")

            tool_calls = message.get("tool_calls")
            if tool_calls:
                result["tool_calls"] = tool_calls

        usage = body.get("usage")
        if usage:
            result["token_usage"] = TokenUsage(
                input_tokens=usage.get("prompt_tokens"),
                output_tokens=usage.get("completion_tokens"),
                total_tokens=usage.get("total_tokens"),
            )

        return result

    def parse_stream_chunk(self, data: str) -> dict[str, Any]:
        """Parse a single SSE data chunk from OpenAI streaming."""
        if data.strip() == "[DONE]":
            return {"finish_reason": "done", "parsed": {"done": True}}

        try:
            parsed = json.loads(data)
        except json.JSONDecodeError:
            return {"parsed": {"raw": data}}

        result: dict[str, Any] = {"parsed": parsed}

        choices = parsed.get("choices", [])
        if choices:
            delta = choices[0].get("delta", {})
            if "content" in delta:
                result["delta_text"] = delta["content"]
            if "tool_calls" in delta:
                result["tool_call_delta"] = delta["tool_calls"]
            finish = choices[0].get("finish_reason")
            if finish:
                result["finish_reason"] = finish

        usage = parsed.get("usage")
        if usage:
            result["token_usage"] = TokenUsage(
                input_tokens=usage.get("prompt_tokens"),
                output_tokens=usage.get("completion_tokens"),
                total_tokens=usage.get("total_tokens"),
            )

        return result

    def reconstruct_response(self, chunks: list[StreamChunk]) -> dict[str, Any]:
        """Reconstruct a full response from OpenAI stream chunks."""
        text_parts: list[str] = []
        tool_calls: dict[int, dict[str, Any]] = {}
        token_usage: TokenUsage | None = None
        model: str | None = None

        for chunk in chunks:
            if chunk.delta_text:
                text_parts.append(chunk.delta_text)
            if chunk.parsed:
                if not model and chunk.parsed.get("model"):
                    model = chunk.parsed["model"]

                # Accumulate tool call deltas
                choices = chunk.parsed.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    for tc in delta.get("tool_calls", []):
                        idx = tc.get("index", 0)
                        if idx not in tool_calls:
                            tool_calls[idx] = {
                                "id": tc.get("id", ""),
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            }
                        if tc.get("id"):
                            tool_calls[idx]["id"] = tc["id"]
                        func = tc.get("function", {})
                        if "name" in func:
                            tool_calls[idx]["function"]["name"] = func["name"]
                        if "arguments" in func:
                            tool_calls[idx]["function"]["arguments"] += func["arguments"]

                usage = chunk.parsed.get("usage")
                if usage:
                    token_usage = TokenUsage(
                        input_tokens=usage.get("prompt_tokens"),
                        output_tokens=usage.get("completion_tokens"),
                        total_tokens=usage.get("total_tokens"),
                    )

        result: dict[str, Any] = {
            "response_text": "".join(text_parts) if text_parts else None,
            "model": model,
        }
        if tool_calls:
            result["tool_calls"] = [tool_calls[i] for i in sorted(tool_calls)]
        if token_usage:
            result["token_usage"] = token_usage

        return result

    def estimate_cost(self, model: str | None, usage: TokenUsage | None) -> CostEstimate | None:
        """Estimate cost for OpenAI models."""
        if not model or not usage:
            return None

        # Find matching pricing (try exact match, then prefix match)
        pricing = OPENAI_PRICING.get(model)
        if not pricing:
            for key, val in OPENAI_PRICING.items():
                if model.startswith(key):
                    pricing = val
                    break

        if not pricing:
            return CostEstimate(model=model, note="Unknown model, no pricing available")

        input_cost = (usage.input_tokens or 0) / 1_000_000 * pricing[0]
        output_cost = (usage.output_tokens or 0) / 1_000_000 * pricing[1]

        return CostEstimate(
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=input_cost + output_cost,
            model=model,
        )
