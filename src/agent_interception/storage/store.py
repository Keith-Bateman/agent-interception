"""Async SQLite store for intercepted interactions."""

from __future__ import annotations

import json
from typing import Any

import aiosqlite

from agent_interception.config import InterceptorConfig
from agent_interception.models import (
    CostEstimate,
    ImageMetadata,
    Interaction,
    Provider,
    StreamChunk,
    TokenUsage,
)
from agent_interception.storage.migrations import apply_migrations


def _serialize_json(value: Any) -> str | None:
    """Serialize a value to JSON string, or None if value is None."""
    if value is None:
        return None
    if isinstance(value, list):
        return json.dumps(
            [
                item.model_dump(mode="json") if hasattr(item, "model_dump") else item
                for item in value
            ]
        )
    if hasattr(value, "model_dump"):
        return json.dumps(value.model_dump(mode="json"))
    return json.dumps(value)


def _deserialize_json(value: str | None) -> Any:
    """Deserialize a JSON string, or None if value is None."""
    if value is None:
        return None
    return json.loads(value)


class InteractionStore:
    """Async SQLite store for saving and querying interactions."""

    def __init__(self, config: InterceptorConfig) -> None:
        self._config = config
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Open the database connection and apply migrations."""
        self._db = await aiosqlite.connect(self._config.db_path)
        self._db.row_factory = aiosqlite.Row
        await apply_migrations(self._db)

    async def close(self) -> None:
        """Close the database connection."""
        if self._db:
            await self._db.close()
            self._db = None

    @property
    def db(self) -> aiosqlite.Connection:
        """Get the database connection, raising if not initialized."""
        if self._db is None:
            raise RuntimeError("Store not initialized. Call initialize() first.")
        return self._db

    async def save(self, interaction: Interaction) -> None:
        """Save an interaction to the database."""
        chunks_json = (
            _serialize_json(interaction.stream_chunks) if self._config.store_stream_chunks else None
        )
        await self.db.execute(
            """
            INSERT OR REPLACE INTO interactions (
                id, session_id, timestamp, method, path, request_headers, request_body,
                raw_request_body, provider, model, system_prompt, messages, tools,
                image_metadata, status_code, response_headers, response_body,
                raw_response_body, is_streaming, stream_chunks, response_text,
                tool_calls, token_usage, cost_estimate, time_to_first_token_ms,
                total_latency_ms, error
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            (
                interaction.id,
                interaction.session_id,
                interaction.timestamp.isoformat(),
                interaction.method,
                interaction.path,
                json.dumps(interaction.request_headers),
                _serialize_json(interaction.request_body),
                interaction.raw_request_body,
                interaction.provider.value,
                interaction.model,
                interaction.system_prompt,
                _serialize_json(interaction.messages),
                _serialize_json(interaction.tools),
                _serialize_json(interaction.image_metadata),
                interaction.status_code,
                json.dumps(interaction.response_headers),
                _serialize_json(interaction.response_body),
                interaction.raw_response_body,
                int(interaction.is_streaming),
                chunks_json,
                interaction.response_text,
                _serialize_json(interaction.tool_calls),
                _serialize_json(interaction.token_usage),
                _serialize_json(interaction.cost_estimate),
                interaction.time_to_first_token_ms,
                interaction.total_latency_ms,
                interaction.error,
            ),
        )
        await self.db.commit()

    async def get(self, interaction_id: str) -> Interaction | None:
        """Get an interaction by ID."""
        cursor = await self.db.execute("SELECT * FROM interactions WHERE id = ?", (interaction_id,))
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_interaction(row)

    async def list_interactions(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        provider: str | None = None,
        model: str | None = None,
        session_id: str | None = None,
    ) -> list[Interaction]:
        """List interactions with optional filtering."""
        query = "SELECT * FROM interactions"
        params: list[Any] = []
        conditions: list[str] = []

        if provider:
            conditions.append("provider = ?")
            params.append(provider)
        if model:
            conditions.append("model = ?")
            params.append(model)
        if session_id:
            conditions.append("session_id = ?")
            params.append(session_id)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor = await self.db.execute(query, params)
        rows = await cursor.fetchall()
        return [self._row_to_interaction(row) for row in rows]

    async def list_sessions(self) -> list[dict[str, Any]]:
        """List all sessions with summary info."""
        cursor = await self.db.execute(
            """
            SELECT
                session_id,
                COUNT(*) as interaction_count,
                MIN(timestamp) as first_interaction,
                MAX(timestamp) as last_interaction,
                GROUP_CONCAT(DISTINCT provider) as providers,
                GROUP_CONCAT(DISTINCT model) as models,
                SUM(total_latency_ms) as total_latency_ms
            FROM interactions
            WHERE session_id IS NOT NULL
            GROUP BY session_id
            ORDER BY first_interaction DESC
            """
        )
        rows = await cursor.fetchall()
        return [
            {
                "session_id": row[0],
                "interaction_count": row[1],
                "first_interaction": row[2],
                "last_interaction": row[3],
                "providers": row[4].split(",") if row[4] else [],
                "models": row[5].split(",") if row[5] else [],
                "total_latency_ms": row[6],
            }
            for row in rows
        ]

    async def clear(self) -> int:
        """Delete all interactions. Returns the number of rows deleted."""
        cursor = await self.db.execute("SELECT COUNT(*) FROM interactions")
        count_row = await cursor.fetchone()
        count = count_row[0] if count_row else 0
        await self.db.execute("DELETE FROM interactions")
        await self.db.commit()
        return count

    async def get_stats(self) -> dict[str, Any]:
        """Get aggregate statistics about stored interactions."""
        cursor = await self.db.execute("SELECT COUNT(*) FROM interactions")
        total_row = await cursor.fetchone()
        total = total_row[0] if total_row else 0

        cursor = await self.db.execute(
            "SELECT provider, COUNT(*) as count FROM interactions GROUP BY provider"
        )
        provider_rows = await cursor.fetchall()
        by_provider = {row[0]: row[1] for row in provider_rows}

        cursor = await self.db.execute(
            "SELECT model, COUNT(*) as count FROM interactions "
            "WHERE model IS NOT NULL GROUP BY model ORDER BY count DESC LIMIT 10"
        )
        model_rows = await cursor.fetchall()
        by_model = {row[0]: row[1] for row in model_rows}

        cursor = await self.db.execute(
            "SELECT AVG(total_latency_ms) FROM interactions WHERE total_latency_ms IS NOT NULL"
        )
        latency_row = await cursor.fetchone()
        avg_latency = latency_row[0] if latency_row else None

        return {
            "total_interactions": total,
            "by_provider": by_provider,
            "by_model": by_model,
            "avg_latency_ms": avg_latency,
        }

    def _row_to_interaction(self, row: aiosqlite.Row) -> Interaction:
        """Convert a database row to an Interaction model."""
        token_usage_data = _deserialize_json(row["token_usage"])
        cost_data = _deserialize_json(row["cost_estimate"])
        image_data = _deserialize_json(row["image_metadata"])
        chunks_data = _deserialize_json(row["stream_chunks"])

        return Interaction(
            id=row["id"],
            session_id=row["session_id"],
            timestamp=row["timestamp"],
            method=row["method"],
            path=row["path"],
            request_headers=json.loads(row["request_headers"]),
            request_body=_deserialize_json(row["request_body"]),
            raw_request_body=row["raw_request_body"],
            provider=Provider(row["provider"]),
            model=row["model"],
            system_prompt=row["system_prompt"],
            messages=_deserialize_json(row["messages"]),
            tools=_deserialize_json(row["tools"]),
            image_metadata=ImageMetadata(**image_data) if image_data else None,
            status_code=row["status_code"],
            response_headers=json.loads(row["response_headers"]),
            response_body=_deserialize_json(row["response_body"]),
            raw_response_body=row["raw_response_body"],
            is_streaming=bool(row["is_streaming"]),
            stream_chunks=[StreamChunk(**c) for c in chunks_data] if chunks_data else [],
            response_text=row["response_text"],
            tool_calls=_deserialize_json(row["tool_calls"]),
            token_usage=TokenUsage(**token_usage_data) if token_usage_data else None,
            cost_estimate=CostEstimate(**cost_data) if cost_data else None,
            time_to_first_token_ms=row["time_to_first_token_ms"],
            total_latency_ms=row["total_latency_ms"],
            error=row["error"],
        )
