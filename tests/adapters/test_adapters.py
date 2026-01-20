"""Tests for database adapters."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_db.adapters.protocol import ConnectionConfig, DatabaseType, SSLConfig, PoolConfig
from agent_db.adapters.factory import create_adapter, create_adapters, DatabasesConfig
from agent_db.adapters.postgresql import PostgreSQLAdapter
from agent_db.adapters.mysql import MySQLAdapter
from agent_db.adapters.qdrant import QdrantAdapter
from agent_db.adapters.milvus import MilvusAdapter
from agent_db.adapters.neo4j import Neo4jAdapter
from agent_db.adapters.nebula import NebulaAdapter
from agent_db.adapters.influxdb import InfluxDBAdapter


class TestConnectionConfig:
    def test_default_port(self):
        config = ConnectionConfig(database_type=DatabaseType.POSTGRESQL)
        assert config.default_port == 5432
        assert config.effective_port == 5432

    def test_custom_port(self):
        config = ConnectionConfig(database_type=DatabaseType.POSTGRESQL, port=5433)
        assert config.effective_port == 5433

    def test_password_secret(self):
        config = ConnectionConfig(
            database_type=DatabaseType.POSTGRESQL,
            password="secret123",
        )
        assert config.get_password() == "secret123"
        # Password should not appear in string representation
        assert "secret123" not in str(config)

    def test_ssl_config(self):
        config = ConnectionConfig(
            database_type=DatabaseType.POSTGRESQL,
            ssl=SSLConfig(enabled=True, verify=False),
        )
        assert config.ssl.enabled is True
        assert config.ssl.verify is False

    def test_pool_config(self):
        config = ConnectionConfig(
            database_type=DatabaseType.MYSQL,
            pool=PoolConfig(min_size=5, max_size=50),
        )
        assert config.pool.min_size == 5
        assert config.pool.max_size == 50


class TestAdapterFactory:
    def test_create_postgresql_adapter(self):
        config = ConnectionConfig(database_type=DatabaseType.POSTGRESQL)
        adapter = create_adapter(config)
        assert isinstance(adapter, PostgreSQLAdapter)

    def test_create_mysql_adapter(self):
        config = ConnectionConfig(database_type=DatabaseType.MYSQL)
        adapter = create_adapter(config)
        assert isinstance(adapter, MySQLAdapter)

    def test_create_qdrant_adapter(self):
        config = ConnectionConfig(database_type=DatabaseType.QDRANT)
        adapter = create_adapter(config)
        assert isinstance(adapter, QdrantAdapter)

    def test_create_milvus_adapter(self):
        config = ConnectionConfig(database_type=DatabaseType.MILVUS)
        adapter = create_adapter(config)
        assert isinstance(adapter, MilvusAdapter)

    def test_create_neo4j_adapter(self):
        config = ConnectionConfig(database_type=DatabaseType.NEO4J)
        adapter = create_adapter(config)
        assert isinstance(adapter, Neo4jAdapter)

    def test_create_nebula_adapter(self):
        config = ConnectionConfig(database_type=DatabaseType.NEBULA)
        adapter = create_adapter(config)
        assert isinstance(adapter, NebulaAdapter)

    def test_create_influxdb_adapter(self):
        config = ConnectionConfig(database_type=DatabaseType.INFLUXDB)
        adapter = create_adapter(config)
        assert isinstance(adapter, InfluxDBAdapter)

    def test_create_adapters_from_config(self):
        configs = DatabasesConfig(databases={
            "pg": ConnectionConfig(database_type=DatabaseType.POSTGRESQL),
            "mysql": ConnectionConfig(database_type=DatabaseType.MYSQL),
        })
        adapters = create_adapters(configs)
        assert len(adapters) == 2
        assert "pg" in adapters
        assert "mysql" in adapters


class TestMySQLAdapter:
    @pytest.fixture
    def config(self) -> ConnectionConfig:
        return ConnectionConfig(
            database_type=DatabaseType.MYSQL,
            host="localhost",
            port=3306,
            database="testdb",
            user="user",
            password="pass",
        )

    def test_adapter_type(self, config: ConnectionConfig):
        adapter = MySQLAdapter(config)
        assert adapter.database_type == DatabaseType.MYSQL

    @pytest.mark.asyncio
    async def test_execute_query(self, config: ConnectionConfig):
        adapter = MySQLAdapter(config)

        # Create mock objects
        mock_cursor = MagicMock()
        mock_cursor.execute = AsyncMock()
        mock_cursor.fetchall = AsyncMock(return_value=[
            {"id": 1, "name": "alice"},
            {"id": 2, "name": "bob"},
        ])
        mock_cursor.__aenter__ = AsyncMock(return_value=mock_cursor)
        mock_cursor.__aexit__ = AsyncMock()

        mock_conn = MagicMock()
        mock_conn.cursor = MagicMock(return_value=mock_cursor)
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock()

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(return_value=mock_conn)

        adapter._pool = mock_pool

        result = await adapter.execute("SELECT * FROM users")
        assert result.row_count == 2


class TestMilvusAdapter:
    @pytest.fixture
    def config(self) -> ConnectionConfig:
        return ConnectionConfig(
            database_type=DatabaseType.MILVUS,
            host="localhost",
            port=19530,
        )

    def test_adapter_type(self, config: ConnectionConfig):
        adapter = MilvusAdapter(config)
        assert adapter.database_type == DatabaseType.MILVUS

    @pytest.mark.asyncio
    async def test_search(self, config: ConnectionConfig):
        adapter = MilvusAdapter(config)

        mock_client = MagicMock()
        mock_client.search.return_value = [[
            {"id": 1, "distance": 0.1, "entity": {"name": "alice"}},
            {"id": 2, "distance": 0.2, "entity": {"name": "bob"}},
        ]]

        adapter._client = mock_client

        result = await adapter.search(
            collection="users",
            vector=[0.1, 0.2, 0.3],
            limit=10,
            output_fields=["name"],
        )
        assert result.row_count == 2


class TestNebulaAdapter:
    @pytest.fixture
    def config(self) -> ConnectionConfig:
        return ConnectionConfig(
            database_type=DatabaseType.NEBULA,
            host="localhost",
            port=9669,
            database="test_space",
            user="root",
            password="nebula",
        )

    def test_adapter_type(self, config: ConnectionConfig):
        adapter = NebulaAdapter(config)
        assert adapter.database_type == DatabaseType.NEBULA
