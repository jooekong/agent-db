"""Database adapter protocol and base types."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional, Union

from pydantic import BaseModel, Field, SecretStr


class DatabaseType(str, Enum):
    """Supported database types."""

    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    QDRANT = "qdrant"
    MILVUS = "milvus"
    NEO4J = "neo4j"
    NEBULA = "nebula"
    INFLUXDB = "influxdb"


class SSLConfig(BaseModel):
    """SSL/TLS configuration."""

    enabled: bool = False
    ca_cert: Optional[str] = None
    client_cert: Optional[str] = None
    client_key: Optional[str] = None
    verify: bool = True


class PoolConfig(BaseModel):
    """Connection pool configuration."""

    min_size: int = 1
    max_size: int = 10
    max_idle_time: int = 300  # seconds
    connect_timeout: int = 10  # seconds
    command_timeout: Optional[int] = None  # seconds


class ConnectionConfig(BaseModel):
    """Database connection configuration."""

    database_type: DatabaseType
    name: str = "default"  # Logical name for this connection
    host: str = "localhost"
    port: Optional[int] = None
    database: Optional[str] = None
    user: Optional[str] = None
    password: Optional[SecretStr] = None
    ssl: SSLConfig = Field(default_factory=SSLConfig)
    pool: PoolConfig = Field(default_factory=PoolConfig)
    extra: dict[str, Any] = Field(default_factory=dict)

    def get_password(self) -> Optional[str]:
        """Get password as plain string."""
        return self.password.get_secret_value() if self.password else None

    @property
    def default_port(self) -> int:
        """Get default port for database type."""
        ports = {
            DatabaseType.POSTGRESQL: 5432,
            DatabaseType.MYSQL: 3306,
            DatabaseType.QDRANT: 6333,
            DatabaseType.MILVUS: 19530,
            DatabaseType.NEO4J: 7687,
            DatabaseType.NEBULA: 9669,
            DatabaseType.INFLUXDB: 8086,
        }
        return ports.get(self.database_type, 0)

    @property
    def effective_port(self) -> int:
        """Get port with fallback to default."""
        return self.port or self.default_port


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
