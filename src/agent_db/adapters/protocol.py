"""Database adapter protocol and base types."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class DatabaseType(str, Enum):
    """Supported database types."""

    POSTGRESQL = "postgresql"
    QDRANT = "qdrant"
    NEO4J = "neo4j"
    INFLUXDB = "influxdb"


class ConnectionConfig(BaseModel):
    """Database connection configuration."""

    database_type: DatabaseType
    host: str = "localhost"
    port: Optional[int] = None
    database: Optional[str] = None
    user: Optional[str] = None
    password: Optional[str] = None
    extra: dict[str, Any] = Field(default_factory=dict)


class QueryResult(BaseModel):
    """Unified query result."""

    columns: list[str]
    rows: list[list[Any]]
    row_count: int
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_dicts(self) -> list[dict[str, Any]]:
        """Convert rows to list of dicts."""
        return [dict(zip(self.columns, row)) for row in self.rows]


class DatabaseAdapter(ABC):
    """Abstract base for database adapters."""

    def __init__(self, config: ConnectionConfig):
        self.config = config

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection."""
        pass

    @abstractmethod
    async def execute(
        self, query: str, params: Optional[dict[str, Any]] = None
    ) -> QueryResult:
        """Execute query and return results."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if connection is healthy."""
        pass

    @property
    @abstractmethod
    def database_type(self) -> DatabaseType:
        """Return database type."""
        pass

    async def __aenter__(self) -> "DatabaseAdapter":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.disconnect()
