"""Tests for the proxy handler."""

from __future__ import annotations

from agent_interception.proxy.handler import redact_headers


class TestRedactHeaders:
    def test_redacts_authorization(self) -> None:
        headers = {"authorization": "Bearer sk-1234567890abcdef"}
        result = redact_headers(headers)
        assert result["authorization"].endswith("***")
        assert "sk-1234567890" not in result["authorization"]

    def test_redacts_api_key(self) -> None:
        headers = {"x-api-key": "sk-ant-api03-verylongkey123"}
        result = redact_headers(headers)
        assert result["x-api-key"].endswith("***")

    def test_preserves_non_sensitive(self) -> None:
        headers = {
            "content-type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        result = redact_headers(headers)
        assert result["content-type"] == "application/json"
        assert result["anthropic-version"] == "2023-06-01"

    def test_no_redaction_when_disabled(self) -> None:
        headers = {"authorization": "Bearer sk-1234567890abcdef"}
        result = redact_headers(headers, redact=False)
        assert result["authorization"] == "Bearer sk-1234567890abcdef"

    def test_short_key_fully_masked(self) -> None:
        headers = {"x-api-key": "short"}
        result = redact_headers(headers)
        assert result["x-api-key"] == "***"
