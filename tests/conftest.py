"""Shared test fixtures."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from typing import Any

import pytest

from agent_interception.config import InterceptorConfig
from agent_interception.models import (
    CostEstimate,
    Interaction,
    Provider,
    StreamChunk,
    TokenUsage,
)
from agent_interception.storage.store import InteractionStore


@pytest.fixture
def config(tmp_path: Any) -> InterceptorConfig:
    """Create a test config with a temporary database."""
    return InterceptorConfig(db_path=str(tmp_path / "test.db"))


@pytest.fixture
async def store(config: InterceptorConfig) -> AsyncGenerator[InteractionStore, None]:
    """Create and initialize a test store."""
    s = InteractionStore(config)
    await s.initialize()
    yield s
    await s.close()


@pytest.fixture
def sample_interaction() -> Interaction:
    """Create a sample interaction for testing."""
    return Interaction(
        id="test-123",
        timestamp=datetime(2025, 1, 15, 10, 30, 0, tzinfo=UTC),
        method="POST",
        path="/v1/messages",
        request_headers={"content-type": "application/json", "authorization": "Bearer sk-***"},
        request_body={
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hello"}],
        },
        provider=Provider.ANTHROPIC,
        model="claude-sonnet-4-20250514",
        system_prompt=None,
        messages=[{"role": "user", "content": "Hello"}],
        tools=None,
        status_code=200,
        response_headers={"content-type": "application/json"},
        response_body={
            "id": "msg_123",
            "content": [{"type": "text", "text": "Hello! How can I help?"}],
        },
        is_streaming=False,
        response_text="Hello! How can I help?",
        token_usage=TokenUsage(input_tokens=10, output_tokens=15),
        cost_estimate=CostEstimate(input_cost=0.00003, output_cost=0.000075, total_cost=0.000105),
        total_latency_ms=450.5,
    )


@pytest.fixture
def sample_streaming_interaction() -> Interaction:
    """Create a sample streaming interaction for testing."""
    return Interaction(
        id="test-stream-456",
        timestamp=datetime(2025, 1, 15, 10, 31, 0, tzinfo=UTC),
        method="POST",
        path="/v1/chat/completions",
        request_headers={"content-type": "application/json"},
        request_body={
            "model": "gpt-4",
            "stream": True,
            "messages": [{"role": "user", "content": "Hi"}],
        },
        provider=Provider.OPENAI,
        model="gpt-4",
        messages=[{"role": "user", "content": "Hi"}],
        status_code=200,
        response_headers={"content-type": "text/event-stream"},
        is_streaming=True,
        stream_chunks=[
            StreamChunk(
                index=0,
                timestamp=datetime(2025, 1, 15, 10, 31, 0, 100000, tzinfo=UTC),
                data='data: {"choices":[{"delta":{"content":"Hello"}}]}',
                delta_text="Hello",
            ),
            StreamChunk(
                index=1,
                timestamp=datetime(2025, 1, 15, 10, 31, 0, 200000, tzinfo=UTC),
                data='data: {"choices":[{"delta":{"content":"!"}}]}',
                delta_text="!",
            ),
        ],
        response_text="Hello!",
        token_usage=TokenUsage(input_tokens=5, output_tokens=2, total_tokens=7),
        time_to_first_token_ms=120.0,
        total_latency_ms=300.0,
    )
