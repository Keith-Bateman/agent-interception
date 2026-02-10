"""Tests for OpenAI provider parser."""

from __future__ import annotations

from datetime import UTC, datetime

from agent_interception.models import StreamChunk
from agent_interception.providers.openai import OpenAIParser


def make_parser() -> OpenAIParser:
    return OpenAIParser()


class TestParseRequest:
    def test_basic_chat_completion(self) -> None:
        parser = make_parser()
        body = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ],
        }
        result = parser.parse_request(body)
        assert result["model"] == "gpt-4o"
        assert result["system_prompt"] == "You are helpful."
        assert len(result["messages"]) == 2
        assert result["is_streaming"] is False
        assert result["tools"] is None

    def test_streaming_request(self) -> None:
        parser = make_parser()
        body = {
            "model": "gpt-4o",
            "stream": True,
            "messages": [{"role": "user", "content": "Hi"}],
        }
        result = parser.parse_request(body)
        assert result["is_streaming"] is True

    def test_with_tools(self) -> None:
        parser = make_parser()
        body = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "What's the weather?"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        }
        result = parser.parse_request(body)
        assert result["tools"] is not None
        assert len(result["tools"]) == 1

    def test_no_system_prompt(self) -> None:
        parser = make_parser()
        body = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hi"}],
        }
        result = parser.parse_request(body)
        assert result["system_prompt"] is None

    def test_image_detection(self) -> None:
        parser = make_parser()
        body = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.com/img.png"},
                        },
                    ],
                }
            ],
        }
        result = parser.parse_request(body)
        assert result["image_metadata"] is not None
        assert result["image_metadata"].count == 1
        assert result["image_metadata"].media_types == ["url"]


class TestParseResponse:
    def test_basic_response(self) -> None:
        parser = make_parser()
        body = {
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }
        result = parser.parse_response(body)
        assert result["response_text"] == "Hello!"
        assert result["token_usage"].input_tokens == 10
        assert result["token_usage"].output_tokens == 5
        assert result["token_usage"].total_tokens == 15

    def test_tool_call_response(self) -> None:
        parser = make_parser()
        body = {
            "model": "gpt-4o",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city":"NYC"}',
                                },
                            }
                        ],
                    }
                }
            ],
        }
        result = parser.parse_response(body)
        assert result["response_text"] is None
        assert result["tool_calls"] is not None
        assert len(result["tool_calls"]) == 1


class TestParseStreamChunk:
    def test_text_delta(self) -> None:
        parser = make_parser()
        data = '{"choices":[{"delta":{"content":"Hello"},"index":0}]}'
        result = parser.parse_stream_chunk(data)
        assert result["delta_text"] == "Hello"

    def test_done_chunk(self) -> None:
        parser = make_parser()
        result = parser.parse_stream_chunk("[DONE]")
        assert result["finish_reason"] == "done"

    def test_usage_in_chunk(self) -> None:
        parser = make_parser()
        data = '{"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}'
        result = parser.parse_stream_chunk(data)
        assert result["token_usage"].total_tokens == 15


class TestReconstructResponse:
    def test_reconstruct_text(self) -> None:
        parser = make_parser()
        now = datetime.now(UTC)
        chunks = [
            StreamChunk(
                index=0,
                timestamp=now,
                data="",
                delta_text="Hello",
                parsed={
                    "choices": [{"delta": {"content": "Hello"}}],
                    "model": "gpt-4o",
                },
            ),
            StreamChunk(
                index=1,
                timestamp=now,
                data="",
                delta_text=" world",
                parsed={"choices": [{"delta": {"content": " world"}}]},
            ),
            StreamChunk(
                index=2,
                timestamp=now,
                data="",
                parsed={
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 5,
                        "completion_tokens": 2,
                        "total_tokens": 7,
                    },
                },
            ),
        ]
        result = parser.reconstruct_response(chunks)
        assert result["response_text"] == "Hello world"
        assert result["model"] == "gpt-4o"
        assert result["token_usage"].total_tokens == 7

    def test_reconstruct_tool_calls(self) -> None:
        parser = make_parser()
        now = datetime.now(UTC)
        chunks = [
            StreamChunk(
                index=0,
                timestamp=now,
                data="",
                parsed={
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_1",
                                        "function": {"name": "get_weather", "arguments": ""},
                                    }
                                ]
                            }
                        }
                    ]
                },
            ),
            StreamChunk(
                index=1,
                timestamp=now,
                data="",
                parsed={
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [{"index": 0, "function": {"arguments": '{"city":'}}]
                            }
                        }
                    ]
                },
            ),
            StreamChunk(
                index=2,
                timestamp=now,
                data="",
                parsed={
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [{"index": 0, "function": {"arguments": '"NYC"}'}}]
                            }
                        }
                    ]
                },
            ),
        ]
        result = parser.reconstruct_response(chunks)
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["function"]["name"] == "get_weather"
        assert result["tool_calls"][0]["function"]["arguments"] == '{"city":"NYC"}'


class TestEstimateCost:
    def test_known_model(self) -> None:
        from agent_interception.models import TokenUsage

        parser = make_parser()
        usage = TokenUsage(input_tokens=1000, output_tokens=500)
        cost = parser.estimate_cost("gpt-4o", usage)
        assert cost is not None
        assert cost.total_cost > 0
        assert cost.model == "gpt-4o"

    def test_unknown_model(self) -> None:
        from agent_interception.models import TokenUsage

        parser = make_parser()
        usage = TokenUsage(input_tokens=100, output_tokens=50)
        cost = parser.estimate_cost("unknown-model", usage)
        assert cost is not None
        assert cost.note is not None
        assert "Unknown" in cost.note

    def test_no_usage(self) -> None:
        parser = make_parser()
        cost = parser.estimate_cost("gpt-4o", None)
        assert cost is None
