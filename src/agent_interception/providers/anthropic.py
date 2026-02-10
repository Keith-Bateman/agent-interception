"""Anthropic request/response/stream parser."""

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
ANTHROPIC_PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-4": (15.00, 75.00),
    "claude-sonnet-4": (3.00, 15.00),
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-3-5-haiku": (0.80, 4.00),
    "claude-3-opus": (15.00, 75.00),
    "claude-3-sonnet": (3.00, 15.00),
    "claude-3-haiku": (0.25, 1.25),
}


class AnthropicParser(ProviderParser):
    """Parser for Anthropic Messages API format."""

    @property
    def provider(self) -> Provider:
        return Provider.ANTHROPIC

    def parse_request(self, body: dict[str, Any]) -> dict[str, Any]:
        """Parse an Anthropic messages request."""
        messages = body.get("messages", [])

        # System prompt can be a string or list of content blocks
        system = body.get("system")
        system_prompt = None
        if isinstance(system, str):
            system_prompt = system
        elif isinstance(system, list):
            text_parts = []
            for block in system:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
            system_prompt = "\n".join(text_parts) if text_parts else None

        return {
            "model": body.get("model"),
            "system_prompt": system_prompt,
            "messages": messages,
            "tools": body.get("tools"),
            "is_streaming": body.get("stream", False),
            "image_metadata": self.extract_image_metadata(messages),
        }

    def parse_response(self, body: dict[str, Any]) -> dict[str, Any]:
        """Parse an Anthropic messages response."""
        result: dict[str, Any] = {"model": body.get("model")}

        content = body.get("content", [])
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []

        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                tool_calls.append(block)
            # Extended thinking blocks
            elif block.get("type") == "thinking":
                text_parts.append(f"[thinking]{block.get('thinking', '')}[/thinking]")

        if text_parts:
            result["response_text"] = "\n".join(text_parts)
        if tool_calls:
            result["tool_calls"] = tool_calls

        usage = body.get("usage")
        if usage:
            result["token_usage"] = TokenUsage(
                input_tokens=usage.get("input_tokens"),
                output_tokens=usage.get("output_tokens"),
                cache_creation_tokens=usage.get("cache_creation_input_tokens"),
                cache_read_tokens=usage.get("cache_read_input_tokens"),
            )

        return result

    def parse_stream_chunk(self, data: str) -> dict[str, Any]:
        """Parse a single SSE data chunk from Anthropic streaming."""
        try:
            parsed = json.loads(data)
        except json.JSONDecodeError:
            return {"parsed": {"raw": data}}

        result: dict[str, Any] = {"parsed": parsed}
        event_type = parsed.get("type", "")

        if event_type == "content_block_delta":
            delta = parsed.get("delta", {})
            delta_type = delta.get("type", "")
            if delta_type == "text_delta":
                result["delta_text"] = delta.get("text", "")
            elif delta_type == "input_json_delta":
                result["tool_call_delta"] = {"partial_json": delta.get("partial_json", "")}
            elif delta_type == "thinking_delta":
                result["delta_text"] = delta.get("thinking", "")

        elif event_type == "message_delta":
            delta = parsed.get("delta", {})
            if delta.get("stop_reason"):
                result["finish_reason"] = delta["stop_reason"]
            usage = parsed.get("usage")
            if usage:
                result["token_usage"] = TokenUsage(
                    output_tokens=usage.get("output_tokens"),
                )

        elif event_type == "message_start":
            message = parsed.get("message", {})
            if message.get("model"):
                result["model"] = message["model"]
            usage = message.get("usage")
            if usage:
                result["token_usage"] = TokenUsage(
                    input_tokens=usage.get("input_tokens"),
                    output_tokens=usage.get("output_tokens"),
                    cache_creation_tokens=usage.get("cache_creation_input_tokens"),
                    cache_read_tokens=usage.get("cache_read_input_tokens"),
                )

        elif event_type == "content_block_start":
            block = parsed.get("content_block", {})
            if block.get("type") == "tool_use":
                result["tool_call_delta"] = {
                    "id": block.get("id"),
                    "name": block.get("name"),
                    "start": True,
                }

        return result

    def reconstruct_response(self, chunks: list[StreamChunk]) -> dict[str, Any]:
        """Reconstruct a full response from Anthropic stream chunks."""
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        current_tool: dict[str, Any] | None = None
        tool_json_parts: list[str] = []
        input_tokens: int | None = None
        output_tokens: int | None = None
        cache_creation_tokens: int | None = None
        cache_read_tokens: int | None = None
        model: str | None = None

        for chunk in chunks:
            if not chunk.parsed:
                continue

            event_type = chunk.parsed.get("type", "")

            if event_type == "message_start":
                message = chunk.parsed.get("message", {})
                model = message.get("model")
                usage = message.get("usage", {})
                input_tokens = usage.get("input_tokens")
                cache_creation_tokens = usage.get("cache_creation_input_tokens")
                cache_read_tokens = usage.get("cache_read_input_tokens")

            elif event_type == "content_block_start":
                block = chunk.parsed.get("content_block", {})
                if block.get("type") == "tool_use":
                    current_tool = {
                        "type": "tool_use",
                        "id": block.get("id", ""),
                        "name": block.get("name", ""),
                    }
                    tool_json_parts = []

            elif event_type == "content_block_delta":
                delta = chunk.parsed.get("delta", {})
                if delta.get("type") == "text_delta":
                    text_parts.append(delta.get("text", ""))
                elif delta.get("type") == "input_json_delta":
                    tool_json_parts.append(delta.get("partial_json", ""))
                elif delta.get("type") == "thinking_delta":
                    text_parts.append(delta.get("thinking", ""))

            elif event_type == "content_block_stop":
                if current_tool is not None:
                    raw_json = "".join(tool_json_parts)
                    try:
                        current_tool["input"] = json.loads(raw_json)
                    except json.JSONDecodeError:
                        current_tool["input"] = raw_json
                    tool_calls.append(current_tool)
                    current_tool = None

            elif event_type == "message_delta":
                usage = chunk.parsed.get("usage", {})
                if "output_tokens" in usage:
                    output_tokens = usage["output_tokens"]

        token_usage = None
        if input_tokens is not None or output_tokens is not None:
            token_usage = TokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_creation_tokens=cache_creation_tokens,
                cache_read_tokens=cache_read_tokens,
            )

        result: dict[str, Any] = {
            "response_text": "".join(text_parts) if text_parts else None,
            "model": model,
        }
        if tool_calls:
            result["tool_calls"] = tool_calls
        if token_usage:
            result["token_usage"] = token_usage

        return result

    def estimate_cost(self, model: str | None, usage: TokenUsage | None) -> CostEstimate | None:
        """Estimate cost for Anthropic models."""
        if not model or not usage:
            return None

        pricing = None
        for key, val in ANTHROPIC_PRICING.items():
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
