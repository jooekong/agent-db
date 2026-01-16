"""Tests for database adapters."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_db.adapters.protocol import ConnectionConfig, DatabaseType
from agent_db.adapters.postgresql import PostgreSQLAdapter
from agent_db.adapters.qdrant import QdrantAdapter
from agent_db.adapters.neo4j import Neo4jAdapter
from agent_db.adapters.influxdb import InfluxDBAdapter


class TestPostgreSQLAdapter:
    @pytest.fixture
    def pg_config(self) -> ConnectionConfig:
        return ConnectionConfig(
            database_type=DatabaseType.POSTGRESQL,
            host="localhost",
            port=5432,
            database="testdb",
            user="user",
            password="pass",
        )

    def test_adapter_init(self, pg_config: ConnectionConfig):
        adapter = PostgreSQLAdapter(pg_config)
        assert adapter.database_type == DatabaseType.POSTGRESQL

    def test_build_dsn(self, pg_config: ConnectionConfig):
        adapter = PostgreSQLAdapter(pg_config)
        dsn = adapter._build_dsn()
        assert "postgresql://" in dsn
        assert "localhost" in dsn

    @pytest.mark.asyncio
    async def test_execute_query(self, pg_config: ConnectionConfig):
        adapter = PostgreSQLAdapter(pg_config)

        mock_pool = MagicMock()
        mock_conn = AsyncMock()

        # Create async context manager mock
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__.return_value = mock_conn
        mock_ctx.__aexit__.return_value = None
        mock_pool.acquire.return_value = mock_ctx

        mock_conn.fetch.return_value = [
            {"id": 1, "name": "alice"},
            {"id": 2, "name": "bob"},
        ]

        adapter._pool = mock_pool

        result = await adapter.execute("SELECT * FROM users")
        assert result.row_count == 2
        assert result.columns == ["id", "name"]


class TestQdrantAdapter:
    @pytest.fixture
    def qdrant_config(self) -> ConnectionConfig:
        return ConnectionConfig(
            database_type=DatabaseType.QDRANT,
            host="localhost",
            port=6333,
        )

    def test_adapter_init(self, qdrant_config: ConnectionConfig):
        adapter = QdrantAdapter(qdrant_config)
        assert adapter.database_type == DatabaseType.QDRANT

    @pytest.mark.asyncio
    async def test_search_vectors(self, qdrant_config: ConnectionConfig):
        adapter = QdrantAdapter(qdrant_config)

        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.id = "1"
        mock_result.score = 0.95
        mock_result.payload = {"name": "alice"}
        mock_client.search.return_value = [mock_result]

        adapter._client = mock_client

        result = await adapter.search(
            collection="users",
            vector=[0.1, 0.2, 0.3],
            limit=10,
        )
        assert result.row_count == 1
        assert "score" in result.columns


class TestNeo4jAdapter:
    @pytest.fixture
    def neo4j_config(self) -> ConnectionConfig:
        return ConnectionConfig(
            database_type=DatabaseType.NEO4J,
            host="localhost",
            port=7687,
            user="neo4j",
            password="password",
        )

    def test_adapter_init(self, neo4j_config: ConnectionConfig):
        adapter = Neo4jAdapter(neo4j_config)
        assert adapter.database_type == DatabaseType.NEO4J

    def test_build_uri(self, neo4j_config: ConnectionConfig):
        adapter = Neo4jAdapter(neo4j_config)
        uri = adapter._build_uri()
        assert uri == "bolt://localhost:7687"


class TestInfluxDBAdapter:
    @pytest.fixture
    def influx_config(self) -> ConnectionConfig:
        return ConnectionConfig(
            database_type=DatabaseType.INFLUXDB,
            host="localhost",
            port=8086,
            extra={
                "token": "test-token",
                "org": "test-org",
            },
        )

    def test_adapter_init(self, influx_config: ConnectionConfig):
        adapter = InfluxDBAdapter(influx_config)
        assert adapter.database_type == DatabaseType.INFLUXDB

    @pytest.mark.asyncio
    async def test_health_check(self, influx_config: ConnectionConfig):
        adapter = InfluxDBAdapter(influx_config)

        mock_client = MagicMock()
        mock_client.ping.return_value = True

        adapter._client = mock_client

        result = await adapter.health_check()
        assert result is True
