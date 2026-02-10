"""Ollama request/response/stream parser (NDJSON format)."""

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


class OllamaParser(ProviderParser):
    """Parser for Ollama API format (NDJSON streaming)."""

    @property
    def provider(self) -> Provider:
        return Provider.OLLAMA

    def parse_request(self, body: dict[str, Any]) -> dict[str, Any]:
        """Parse an Ollama request (/api/chat or /api/generate)."""
        model = body.get("model")

        # /api/chat format
        messages = body.get("messages")
        system_prompt = body.get("system")

        if messages:
            for msg in messages:
                if msg.get("role") == "system":
                    system_prompt = msg.get("content")
                    break

        # /api/generate format uses "prompt" instead of "messages"
        prompt = body.get("prompt")
        if prompt and not messages:
            messages = [{"role": "user", "content": prompt}]

        # stream defaults to True for Ollama
        is_streaming = body.get("stream", True)

        return {
            "model": model,
            "system_prompt": system_prompt,
            "messages": messages,
            "tools": body.get("tools"),
            "is_streaming": is_streaming,
            "image_metadata": None,
        }

    def parse_response(self, body: dict[str, Any]) -> dict[str, Any]:
        """Parse an Ollama non-streaming response."""
        result: dict[str, Any] = {"model": body.get("model")}

        # /api/chat format
        message = body.get("message")
        if message:
            result["response_text"] = message.get("content")
            tool_calls = message.get("tool_calls")
            if tool_calls:
                result["tool_calls"] = tool_calls

        # /api/generate format
        if "response" in body:
            result["response_text"] = body["response"]

        # Token counts
        input_tokens = body.get("prompt_eval_count")
        output_tokens = body.get("eval_count")
        if input_tokens is not None or output_tokens is not None:
            result["token_usage"] = TokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        return result

    def parse_stream_chunk(self, data: str) -> dict[str, Any]:
        """Parse a single NDJSON line from Ollama streaming."""
        try:
            parsed = json.loads(data)
        except json.JSONDecodeError:
            return {"parsed": {"raw": data}}

        result: dict[str, Any] = {"parsed": parsed}

        # /api/chat streaming
        message = parsed.get("message")
        if message:
            content = message.get("content")
            if content:
                result["delta_text"] = content

        # /api/generate streaming
        if "response" in parsed:
            result["delta_text"] = parsed["response"]

        if parsed.get("done"):
            result["finish_reason"] = "done"

            input_tokens = parsed.get("prompt_eval_count")
            output_tokens = parsed.get("eval_count")
            if input_tokens is not None or output_tokens is not None:
                result["token_usage"] = TokenUsage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )

        return result

    def reconstruct_response(self, chunks: list[StreamChunk]) -> dict[str, Any]:
        """Reconstruct a full response from Ollama NDJSON stream chunks."""
        text_parts: list[str] = []
        token_usage: TokenUsage | None = None
        model: str | None = None

        for chunk in chunks:
            if chunk.delta_text:
                text_parts.append(chunk.delta_text)
            if chunk.parsed:
                if not model and chunk.parsed.get("model"):
                    model = chunk.parsed["model"]
                if chunk.parsed.get("done"):
                    input_tokens = chunk.parsed.get("prompt_eval_count")
                    output_tokens = chunk.parsed.get("eval_count")
                    if input_tokens is not None or output_tokens is not None:
                        token_usage = TokenUsage(
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                        )

        return {
            "response_text": "".join(text_parts) if text_parts else None,
            "model": model,
            "token_usage": token_usage,
        }

    def estimate_cost(self, model: str | None, usage: TokenUsage | None) -> CostEstimate | None:
        """Ollama runs locally, so cost is always zero."""
        if not model:
            return None
        return CostEstimate(
            input_cost=0.0,
            output_cost=0.0,
            total_cost=0.0,
            model=model,
            note="Local model (Ollama) - no API cost",
        )
