"""Tests for Anthropic provider parser."""

from __future__ import annotations

from datetime import UTC, datetime

from agent_interception.models import StreamChunk, TokenUsage
from agent_interception.providers.anthropic import AnthropicParser


def make_parser() -> AnthropicParser:
    return AnthropicParser()


class TestParseRequest:
    def test_basic_message(self) -> None:
        parser = make_parser()
        body = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hello"}],
        }
        result = parser.parse_request(body)
        assert result["model"] == "claude-sonnet-4-20250514"
        assert result["system_prompt"] is None
        assert len(result["messages"]) == 1

    def test_with_string_system(self) -> None:
        parser = make_parser()
        body = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "system": "You are helpful.",
            "messages": [{"role": "user", "content": "Hi"}],
        }
        result = parser.parse_request(body)
        assert result["system_prompt"] == "You are helpful."

    def test_with_block_system(self) -> None:
        parser = make_parser()
        body = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "system": [
                {"type": "text", "text": "You are helpful."},
                {"type": "text", "text": "Be concise."},
            ],
            "messages": [{"role": "user", "content": "Hi"}],
        }
        result = parser.parse_request(body)
        assert result["system_prompt"] == "You are helpful.\nBe concise."

    def test_streaming(self) -> None:
        parser = make_parser()
        body = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "stream": True,
            "messages": [{"role": "user", "content": "Hi"}],
        }
        result = parser.parse_request(body)
        assert result["is_streaming"] is True

    def test_with_tools(self) -> None:
        parser = make_parser()
        body = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "What's the weather?"}],
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Get weather info",
                    "input_schema": {"type": "object", "properties": {}},
                }
            ],
        }
        result = parser.parse_request(body)
        assert result["tools"] is not None
        assert len(result["tools"]) == 1

    def test_image_detection(self) -> None:
        parser = make_parser()
        body = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image."},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "iVBORw0KGgo=",
                            },
                        },
                    ],
                }
            ],
        }
        result = parser.parse_request(body)
        assert result["image_metadata"] is not None
        assert result["image_metadata"].count == 1
        assert result["image_metadata"].media_types == ["image/png"]


class TestParseResponse:
    def test_text_response(self) -> None:
        parser = make_parser()
        body = {
            "id": "msg_123",
            "model": "claude-sonnet-4-20250514",
            "content": [{"type": "text", "text": "Hello! How can I help?"}],
            "usage": {"input_tokens": 10, "output_tokens": 8},
        }
        result = parser.parse_response(body)
        assert result["response_text"] == "Hello! How can I help?"
        assert result["token_usage"].input_tokens == 10
        assert result["token_usage"].output_tokens == 8

    def test_tool_use_response(self) -> None:
        parser = make_parser()
        body = {
            "model": "claude-sonnet-4-20250514",
            "content": [
                {"type": "text", "text": "I'll check the weather."},
                {
                    "type": "tool_use",
                    "id": "toolu_123",
                    "name": "get_weather",
                    "input": {"city": "NYC"},
                },
            ],
            "usage": {"input_tokens": 20, "output_tokens": 15},
        }
        result = parser.parse_response(body)
        assert "I'll check the weather." in result["response_text"]
        assert result["tool_calls"] is not None
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["name"] == "get_weather"

    def test_cache_tokens(self) -> None:
        parser = make_parser()
        body = {
            "model": "claude-sonnet-4-20250514",
            "content": [{"type": "text", "text": "Hello"}],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "cache_creation_input_tokens": 100,
                "cache_read_input_tokens": 50,
            },
        }
        result = parser.parse_response(body)
        assert result["token_usage"].cache_creation_tokens == 100
        assert result["token_usage"].cache_read_tokens == 50


class TestParseStreamChunk:
    def test_text_delta(self) -> None:
        parser = make_parser()
        data = (
            '{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}'
        )
        result = parser.parse_stream_chunk(data)
        assert result["delta_text"] == "Hello"

    def test_message_start(self) -> None:
        parser = make_parser()
        data = (
            '{"type":"message_start","message":'
            '{"model":"claude-sonnet-4-20250514",'
            '"usage":{"input_tokens":10,"output_tokens":0}}}'
        )
        result = parser.parse_stream_chunk(data)
        assert result["model"] == "claude-sonnet-4-20250514"
        assert result["token_usage"].input_tokens == 10

    def test_message_delta_with_stop(self) -> None:
        parser = make_parser()
        data = (
            '{"type":"message_delta",'
            '"delta":{"stop_reason":"end_turn"},'
            '"usage":{"output_tokens":15}}'
        )
        result = parser.parse_stream_chunk(data)
        assert result["finish_reason"] == "end_turn"
        assert result["token_usage"].output_tokens == 15

    def test_tool_use_start(self) -> None:
        parser = make_parser()
        data = (
            '{"type":"content_block_start",'
            '"index":1,"content_block":'
            '{"type":"tool_use",'
            '"id":"toolu_123","name":"get_weather"}}'
        )
        result = parser.parse_stream_chunk(data)
        assert result["tool_call_delta"]["id"] == "toolu_123"
        assert result["tool_call_delta"]["name"] == "get_weather"

    def test_input_json_delta(self) -> None:
        parser = make_parser()
        data = (
            '{"type":"content_block_delta",'
            '"index":1,"delta":'
            '{"type":"input_json_delta",'
            '"partial_json":"{\\"city\\":"}}'
        )
        result = parser.parse_stream_chunk(data)
        assert result["tool_call_delta"]["partial_json"] == '{"city":'

    def test_thinking_delta(self) -> None:
        parser = make_parser()
        data = (
            '{"type":"content_block_delta",'
            '"index":0,"delta":'
            '{"type":"thinking_delta",'
            '"thinking":"Let me think..."}}'
        )
        result = parser.parse_stream_chunk(data)
        assert result["delta_text"] == "Let me think..."


class TestReconstructResponse:
    def test_reconstruct_text(self) -> None:
        parser = make_parser()
        now = datetime.now(UTC)
        chunks = [
            StreamChunk(
                index=0,
                timestamp=now,
                data="",
                parsed={
                    "type": "message_start",
                    "message": {
                        "model": "claude-sonnet-4-20250514",
                        "usage": {"input_tokens": 10, "output_tokens": 0},
                    },
                },
            ),
            StreamChunk(
                index=1,
                timestamp=now,
                data="",
                delta_text="Hello",
                parsed={
                    "type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": "Hello"},
                },
            ),
            StreamChunk(
                index=2,
                timestamp=now,
                data="",
                delta_text=" world",
                parsed={
                    "type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": " world"},
                },
            ),
            StreamChunk(
                index=3,
                timestamp=now,
                data="",
                parsed={
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn"},
                    "usage": {"output_tokens": 5},
                },
            ),
        ]
        result = parser.reconstruct_response(chunks)
        assert result["response_text"] == "Hello world"
        assert result["model"] == "claude-sonnet-4-20250514"
        assert result["token_usage"].input_tokens == 10
        assert result["token_usage"].output_tokens == 5

    def test_reconstruct_tool_use(self) -> None:
        parser = make_parser()
        now = datetime.now(UTC)
        chunks = [
            StreamChunk(
                index=0,
                timestamp=now,
                data="",
                parsed={
                    "type": "message_start",
                    "message": {
                        "model": "claude-sonnet-4-20250514",
                        "usage": {"input_tokens": 20, "output_tokens": 0},
                    },
                },
            ),
            StreamChunk(
                index=1,
                timestamp=now,
                data="",
                parsed={
                    "type": "content_block_start",
                    "content_block": {"type": "tool_use", "id": "toolu_1", "name": "get_weather"},
                },
            ),
            StreamChunk(
                index=2,
                timestamp=now,
                data="",
                parsed={
                    "type": "content_block_delta",
                    "delta": {"type": "input_json_delta", "partial_json": '{"city":'},
                },
            ),
            StreamChunk(
                index=3,
                timestamp=now,
                data="",
                parsed={
                    "type": "content_block_delta",
                    "delta": {"type": "input_json_delta", "partial_json": '"NYC"}'},
                },
            ),
            StreamChunk(
                index=4,
                timestamp=now,
                data="",
                parsed={"type": "content_block_stop", "index": 1},
            ),
            StreamChunk(
                index=5,
                timestamp=now,
                data="",
                parsed={
                    "type": "message_delta",
                    "delta": {"stop_reason": "tool_use"},
                    "usage": {"output_tokens": 10},
                },
            ),
        ]
        result = parser.reconstruct_response(chunks)
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["name"] == "get_weather"
        assert result["tool_calls"][0]["input"] == {"city": "NYC"}


class TestEstimateCost:
    def test_known_model(self) -> None:
        parser = make_parser()
        usage = TokenUsage(input_tokens=1000, output_tokens=500)
        cost = parser.estimate_cost("claude-sonnet-4-20250514", usage)
        assert cost is not None
        assert cost.total_cost > 0

    def test_unknown_model(self) -> None:
        parser = make_parser()
        usage = TokenUsage(input_tokens=100, output_tokens=50)
        cost = parser.estimate_cost("claude-unknown-model", usage)
        assert cost is not None
        assert "Unknown" in (cost.note or "")
