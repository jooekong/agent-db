"""Database adapters for unified data access."""

from agent_db.adapters.protocol import (
    DatabaseAdapter,
    QueryResult,
    ConnectionConfig,
    DatabaseType,
)
from agent_db.adapters.postgresql import PostgreSQLAdapter
from agent_db.adapters.qdrant import QdrantAdapter
from agent_db.adapters.neo4j import Neo4jAdapter
from agent_db.adapters.influxdb import InfluxDBAdapter

__all__ = [
    "DatabaseAdapter",
    "QueryResult",
    "ConnectionConfig",
    "DatabaseType",
    "PostgreSQLAdapter",
    "QdrantAdapter",
    "Neo4jAdapter",
    "InfluxDBAdapter",
]
