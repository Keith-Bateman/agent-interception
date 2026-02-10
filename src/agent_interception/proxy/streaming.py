"""SSE and NDJSON stream interception."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

from agent_interception.models import Provider, StreamChunk
from agent_interception.providers.base import ProviderParser


class StreamInterceptor:
    """Intercepts a streaming response, forwarding raw bytes while buffering for parsing.

    The agent receives exactly what the provider sent (byte-level fidelity).
    We parse a buffered copy for logging.
    """

    def __init__(self, parser: ProviderParser, provider: Provider) -> None:
        self._parser = parser
        self._provider = provider
        self._chunks: list[StreamChunk] = []
        self._chunk_index = 0
        self._first_chunk_time: datetime | None = None
        self._buffer = ""

    @property
    def chunks(self) -> list[StreamChunk]:
        """Return all parsed chunks."""
        return self._chunks

    @property
    def first_chunk_time(self) -> datetime | None:
        """Return the timestamp of the first received chunk."""
        return self._first_chunk_time

    async def intercept(self, raw_stream: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
        """Yield raw bytes from the stream while parsing a buffered copy.

        For SSE (OpenAI, Anthropic): parse `data:` lines.
        For NDJSON (Ollama): parse each line as JSON.
        """
        is_ndjson = self._provider == Provider.OLLAMA

        async for raw_bytes in raw_stream:
            now = datetime.now(UTC)
            if self._first_chunk_time is None:
                self._first_chunk_time = now

            # Forward raw bytes immediately
            yield raw_bytes

            # Buffer for line-based parsing
            text = raw_bytes.decode("utf-8", errors="replace")
            self._buffer += text

            # Process complete lines
            while "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
                line = line.strip()
                if not line:
                    continue

                if is_ndjson:
                    self._parse_ndjson_line(line, now)
                else:
                    self._parse_sse_line(line, now)

    def _parse_sse_line(self, line: str, timestamp: datetime) -> None:
        """Parse an SSE line (data: prefix)."""
        if not line.startswith("data:"):
            # Ignore event:, id:, retry:, etc.
            return

        data = line[5:].strip()
        if not data:
            return

        parsed_result = self._parser.parse_stream_chunk(data)
        chunk = StreamChunk(
            index=self._chunk_index,
            timestamp=timestamp,
            data=line,
            parsed=parsed_result.get("parsed"),
            delta_text=parsed_result.get("delta_text"),
        )
        self._chunks.append(chunk)
        self._chunk_index += 1

    def _parse_ndjson_line(self, line: str, timestamp: datetime) -> None:
        """Parse an NDJSON line."""
        parsed_result = self._parser.parse_stream_chunk(line)
        chunk = StreamChunk(
            index=self._chunk_index,
            timestamp=timestamp,
            data=line,
            parsed=parsed_result.get("parsed"),
            delta_text=parsed_result.get("delta_text"),
        )
        self._chunks.append(chunk)
        self._chunk_index += 1


def should_inject_stream_options(body: dict[str, Any], provider: Provider) -> bool:
    """Check if we should inject stream_options.include_usage for OpenAI."""
    if provider != Provider.OPENAI:
        return False
    if not body.get("stream"):
        return False
    stream_options = body.get("stream_options", {})
    return not stream_options.get("include_usage", False)


def inject_stream_options(body: dict[str, Any]) -> dict[str, Any]:
    """Inject stream_options.include_usage into an OpenAI request body."""
    body = body.copy()
    stream_options = body.get("stream_options", {})
    stream_options["include_usage"] = True
    body["stream_options"] = stream_options
    return body
