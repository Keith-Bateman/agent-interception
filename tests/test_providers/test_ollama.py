"""Tests for Ollama provider parser."""

from __future__ import annotations

from datetime import UTC, datetime

from agent_interception.models import StreamChunk
from agent_interception.providers.ollama import OllamaParser


def make_parser() -> OllamaParser:
    return OllamaParser()


class TestParseRequest:
    def test_chat_request(self) -> None:
        parser = make_parser()
        body = {
            "model": "llama3.2",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ],
        }
        result = parser.parse_request(body)
        assert result["model"] == "llama3.2"
        assert result["system_prompt"] == "You are helpful."
        assert len(result["messages"]) == 2
        assert result["is_streaming"] is True  # default for Ollama

    def test_generate_request(self) -> None:
        parser = make_parser()
        body = {
            "model": "llama3.2",
            "prompt": "Why is the sky blue?",
            "system": "Be scientific.",
        }
        result = parser.parse_request(body)
        assert result["model"] == "llama3.2"
        assert result["system_prompt"] == "Be scientific."
        assert result["messages"] == [{"role": "user", "content": "Why is the sky blue?"}]

    def test_non_streaming(self) -> None:
        parser = make_parser()
        body = {
            "model": "llama3.2",
            "stream": False,
            "messages": [{"role": "user", "content": "Hi"}],
        }
        result = parser.parse_request(body)
        assert result["is_streaming"] is False


class TestParseResponse:
    def test_chat_response(self) -> None:
        parser = make_parser()
        body = {
            "model": "llama3.2",
            "message": {"role": "assistant", "content": "Hello!"},
            "done": True,
            "prompt_eval_count": 10,
            "eval_count": 5,
        }
        result = parser.parse_response(body)
        assert result["response_text"] == "Hello!"
        assert result["token_usage"].input_tokens == 10
        assert result["token_usage"].output_tokens == 5

    def test_generate_response(self) -> None:
        parser = make_parser()
        body = {
            "model": "llama3.2",
            "response": "The sky is blue because...",
            "done": True,
            "prompt_eval_count": 8,
            "eval_count": 20,
        }
        result = parser.parse_response(body)
        assert result["response_text"] == "The sky is blue because..."


class TestParseStreamChunk:
    def test_chat_chunk(self) -> None:
        parser = make_parser()
        data = '{"model":"llama3.2","message":{"role":"assistant","content":"Hi"},"done":false}'
        result = parser.parse_stream_chunk(data)
        assert result["delta_text"] == "Hi"

    def test_generate_chunk(self) -> None:
        parser = make_parser()
        data = '{"model":"llama3.2","response":"The","done":false}'
        result = parser.parse_stream_chunk(data)
        assert result["delta_text"] == "The"

    def test_done_chunk(self) -> None:
        parser = make_parser()
        data = '{"model":"llama3.2","done":true,"prompt_eval_count":10,"eval_count":50}'
        result = parser.parse_stream_chunk(data)
        assert result["finish_reason"] == "done"
        assert result["token_usage"].input_tokens == 10
        assert result["token_usage"].output_tokens == 50


class TestReconstructResponse:
    def test_reconstruct(self) -> None:
        parser = make_parser()
        now = datetime.now(UTC)
        chunks = [
            StreamChunk(
                index=0,
                timestamp=now,
                data="",
                delta_text="Hello",
                parsed={"model": "llama3.2", "message": {"content": "Hello"}, "done": False},
            ),
            StreamChunk(
                index=1,
                timestamp=now,
                data="",
                delta_text=" there",
                parsed={"model": "llama3.2", "message": {"content": " there"}, "done": False},
            ),
            StreamChunk(
                index=2,
                timestamp=now,
                data="",
                parsed={
                    "model": "llama3.2",
                    "done": True,
                    "prompt_eval_count": 5,
                    "eval_count": 2,
                },
            ),
        ]
        result = parser.reconstruct_response(chunks)
        assert result["response_text"] == "Hello there"
        assert result["model"] == "llama3.2"
        assert result["token_usage"].input_tokens == 5
        assert result["token_usage"].output_tokens == 2


class TestEstimateCost:
    def test_always_free(self) -> None:
        from agent_interception.models import TokenUsage

        parser = make_parser()
        usage = TokenUsage(input_tokens=1000, output_tokens=500)
        cost = parser.estimate_cost("llama3.2", usage)
        assert cost is not None
        assert cost.total_cost == 0.0
        assert "Local" in (cost.note or "")
