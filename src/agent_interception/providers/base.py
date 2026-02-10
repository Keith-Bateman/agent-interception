"""Abstract base class for provider parsers."""

from __future__ import annotations

import base64
from abc import ABC, abstractmethod
from typing import Any

from agent_interception.models import (
    CostEstimate,
    ImageMetadata,
    Provider,
    StreamChunk,
    TokenUsage,
)


class ProviderParser(ABC):
    """Abstract interface for parsing provider-specific request/response formats."""

    @property
    @abstractmethod
    def provider(self) -> Provider:
        """Return the provider enum value."""

    @abstractmethod
    def parse_request(self, body: dict[str, Any]) -> dict[str, Any]:
        """Parse a request body into normalized fields.

        Returns a dict with optional keys:
        - model: str
        - system_prompt: str | None
        - messages: list[dict]
        - tools: list[dict] | None
        - is_streaming: bool
        - image_metadata: ImageMetadata | None
        """

    @abstractmethod
    def parse_response(self, body: dict[str, Any]) -> dict[str, Any]:
        """Parse a non-streaming response body into normalized fields.

        Returns a dict with optional keys:
        - response_text: str | None
        - tool_calls: list[dict] | None
        - token_usage: TokenUsage | None
        - model: str | None (if model is in response but not request)
        """

    @abstractmethod
    def parse_stream_chunk(self, data: str) -> dict[str, Any]:
        """Parse a single stream chunk.

        Returns a dict with optional keys:
        - delta_text: str | None
        - tool_call_delta: dict | None
        - token_usage: TokenUsage | None
        - finish_reason: str | None
        - parsed: dict (the raw parsed chunk)
        """

    @abstractmethod
    def reconstruct_response(self, chunks: list[StreamChunk]) -> dict[str, Any]:
        """Reconstruct a full response from stream chunks.

        Returns a dict with the same keys as parse_response.
        """

    def estimate_cost(self, model: str | None, usage: TokenUsage | None) -> CostEstimate | None:
        """Estimate cost based on model and token usage. Override in subclasses."""
        return None

    @staticmethod
    def extract_image_metadata(messages: list[dict[str, Any]]) -> ImageMetadata | None:
        """Extract image metadata from messages without storing raw base64."""
        count = 0
        media_types: list[str] = []
        sizes: list[int] = []

        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                # OpenAI format
                if part.get("type") == "image_url":
                    url = part.get("image_url", {}).get("url", "")
                    count += 1
                    if url.startswith("data:"):
                        media_type = url.split(";")[0].split(":")[1] if ";" in url else "unknown"
                        media_types.append(media_type)
                        b64_data = url.split(",", 1)[1] if "," in url else ""
                        sizes.append(len(base64.b64decode(b64_data)) if b64_data else 0)
                    else:
                        media_types.append("url")
                        sizes.append(0)
                # Anthropic format
                elif part.get("type") == "image":
                    source = part.get("source", {})
                    count += 1
                    media_types.append(source.get("media_type", "unknown"))
                    b64_data = source.get("data", "")
                    sizes.append(len(base64.b64decode(b64_data)) if b64_data else 0)

        if count == 0:
            return None
        return ImageMetadata(count=count, media_types=media_types, approximate_sizes=sizes)
