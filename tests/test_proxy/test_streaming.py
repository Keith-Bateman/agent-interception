"""Tests for stream interception."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from agent_interception.models import Provider
from agent_interception.providers.anthropic import AnthropicParser
from agent_interception.providers.ollama import OllamaParser
from agent_interception.providers.openai import OpenAIParser
from agent_interception.proxy.streaming import (
    StreamInterceptor,
    inject_stream_options,
    should_inject_stream_options,
)


async def _make_stream(lines: list[str]) -> AsyncIterator[bytes]:
    """Create an async iterator of bytes from lines."""
    for line in lines:
        yield (line + "\n").encode("utf-8")


@pytest.mark.asyncio
async def test_sse_interception_openai() -> None:
    """Test SSE stream interception for OpenAI format."""
    parser = OpenAIParser()
    interceptor = StreamInterceptor(parser, Provider.OPENAI)

    lines = [
        'data: {"choices":[{"delta":{"content":"Hello"}}]}',
        "",
        'data: {"choices":[{"delta":{"content":" world"}}]}',
        "",
        "data: [DONE]",
    ]

    collected: list[bytes] = []
    async for chunk in interceptor.intercept(_make_stream(lines)):
        collected.append(chunk)

    # All bytes forwarded
    assert len(collected) == len(lines)

    # Chunks parsed
    assert len(interceptor.chunks) == 3
    assert interceptor.chunks[0].delta_text == "Hello"
    assert interceptor.chunks[1].delta_text == " world"
    assert interceptor.first_chunk_time is not None


@pytest.mark.asyncio
async def test_sse_interception_anthropic() -> None:
    """Test SSE stream interception for Anthropic format."""
    parser = AnthropicParser()
    interceptor = StreamInterceptor(parser, Provider.ANTHROPIC)

    lines = [
        "event: message_start",
        (
            'data: {"type":"message_start","message":'
            '{"model":"claude-sonnet-4-20250514",'
            '"usage":{"input_tokens":10,"output_tokens":0}}}'
        ),
        "",
        "event: content_block_delta",
        'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hi"}}',
        "",
        "event: message_delta",
        (
            'data: {"type":"message_delta",'
            '"delta":{"stop_reason":"end_turn"},'
            '"usage":{"output_tokens":1}}'
        ),
        "",
    ]

    collected: list[bytes] = []
    async for chunk in interceptor.intercept(_make_stream(lines)):
        collected.append(chunk)

    # Only data: lines are parsed (not event: lines)
    assert len(interceptor.chunks) == 3
    assert interceptor.chunks[1].delta_text == "Hi"


@pytest.mark.asyncio
async def test_ndjson_interception_ollama() -> None:
    """Test NDJSON stream interception for Ollama format."""
    parser = OllamaParser()
    interceptor = StreamInterceptor(parser, Provider.OLLAMA)

    lines = [
        '{"model":"llama3.2","message":{"role":"assistant","content":"Hello"},"done":false}',
        '{"model":"llama3.2","message":{"role":"assistant","content":"!"},"done":false}',
        '{"model":"llama3.2","done":true,"prompt_eval_count":5,"eval_count":2}',
    ]

    collected: list[bytes] = []
    async for chunk in interceptor.intercept(_make_stream(lines)):
        collected.append(chunk)

    assert len(interceptor.chunks) == 3
    assert interceptor.chunks[0].delta_text == "Hello"
    assert interceptor.chunks[1].delta_text == "!"


class TestStreamOptions:
    def test_should_inject_openai_streaming(self) -> None:
        body = {"model": "gpt-4o", "stream": True, "messages": []}
        assert should_inject_stream_options(body, Provider.OPENAI) is True

    def test_should_not_inject_non_streaming(self) -> None:
        body = {"model": "gpt-4o", "stream": False, "messages": []}
        assert should_inject_stream_options(body, Provider.OPENAI) is False

    def test_should_not_inject_already_present(self) -> None:
        body = {
            "model": "gpt-4o",
            "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [],
        }
        assert should_inject_stream_options(body, Provider.OPENAI) is False

    def test_should_not_inject_anthropic(self) -> None:
        body = {"model": "claude-sonnet-4-20250514", "stream": True, "messages": []}
        assert should_inject_stream_options(body, Provider.ANTHROPIC) is False

    def test_inject(self) -> None:
        body = {"model": "gpt-4o", "stream": True, "messages": []}
        modified = inject_stream_options(body)
        assert modified["stream_options"]["include_usage"] is True
        # Original not modified
        assert "stream_options" not in body
