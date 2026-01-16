"""Database adapters for unified data access."""

from agent_db.adapters.protocol import (
    DatabaseAdapter,
    QueryResult,
    ConnectionConfig,
    DatabaseType,
)

__all__ = [
    "DatabaseAdapter",
    "QueryResult",
    "ConnectionConfig",
    "DatabaseType",
]
