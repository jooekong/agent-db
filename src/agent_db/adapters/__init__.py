"""Database adapters for unified data access."""

from agent_db.adapters.protocol import (
    DatabaseAdapter,
    QueryResult,
    ConnectionConfig,
    DatabaseType,
    SSLConfig,
    PoolConfig,
)
from agent_db.adapters.factory import (
    DatabasesConfig,
    create_adapter,
    create_adapters,
    connect_all,
    disconnect_all,
    register_adapter,
)

# Import all adapters to trigger registration
from agent_db.adapters.postgresql import PostgreSQLAdapter
from agent_db.adapters.mysql import MySQLAdapter
from agent_db.adapters.qdrant import QdrantAdapter
from agent_db.adapters.milvus import MilvusAdapter
from agent_db.adapters.neo4j import Neo4jAdapter
from agent_db.adapters.nebula import NebulaAdapter
from agent_db.adapters.influxdb import InfluxDBAdapter

__all__ = [
    # Protocol
    "DatabaseAdapter",
    "QueryResult",
    "ConnectionConfig",
    "DatabaseType",
    "SSLConfig",
    "PoolConfig",
    # Factory
    "DatabasesConfig",
    "create_adapter",
    "create_adapters",
    "connect_all",
    "disconnect_all",
    "register_adapter",
    # Adapters
    "PostgreSQLAdapter",
    "MySQLAdapter",
    "QdrantAdapter",
    "MilvusAdapter",
    "Neo4jAdapter",
    "NebulaAdapter",
    "InfluxDBAdapter",
]
