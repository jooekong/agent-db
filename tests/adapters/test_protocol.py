"""Tests for database adapter protocol."""

from typing import Any

import pytest

from agent_db.adapters.protocol import (
    DatabaseAdapter,
    QueryResult,
    ConnectionConfig,
    DatabaseType,
)


class MockAdapter(DatabaseAdapter):
    """Mock adapter for testing protocol."""

    async def connect(self) -> None:
        pass

    async def disconnect(self) -> None:
        pass

    async def execute(self, query: str, params: dict[str, Any] | None = None) -> QueryResult:
        return QueryResult(columns=["id"], rows=[[1]], row_count=1)

    async def health_check(self) -> bool:
        return True

    @property
    def database_type(self) -> DatabaseType:
        return DatabaseType.POSTGRESQL


class TestQueryResult:
    def test_result_basic(self):
        result = QueryResult(
            columns=["id", "name"],
            rows=[[1, "alice"], [2, "bob"]],
            row_count=2,
        )
        assert result.row_count == 2
        assert len(result.columns) == 2

    def test_result_to_dicts(self):
        result = QueryResult(
            columns=["id", "name"],
            rows=[[1, "alice"], [2, "bob"]],
            row_count=2,
        )
        dicts = result.to_dicts()
        assert len(dicts) == 2
        assert dicts[0] == {"id": 1, "name": "alice"}


class TestConnectionConfig:
    def test_config_basic(self):
        config = ConnectionConfig(
            database_type=DatabaseType.POSTGRESQL,
            host="localhost",
            port=5432,
            database="testdb",
            user="user",
            password="pass",
        )
        assert config.host == "localhost"
        assert config.port == 5432


class TestMockAdapter:
    @pytest.mark.asyncio
    async def test_adapter_execute(self):
        adapter = MockAdapter(ConnectionConfig(
            database_type=DatabaseType.POSTGRESQL,
            host="localhost",
        ))
        result = await adapter.execute("SELECT 1")
        assert result.row_count == 1
