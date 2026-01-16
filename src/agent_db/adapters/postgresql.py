"""PostgreSQL database adapter."""

from typing import Any, Optional

import asyncpg

from agent_db.adapters.protocol import (
    DatabaseAdapter,
    ConnectionConfig,
    DatabaseType,
    QueryResult,
)


class PostgreSQLAdapter(DatabaseAdapter):
    """Adapter for PostgreSQL databases."""

    def __init__(self, config: ConnectionConfig):
        super().__init__(config)
        self._pool: Optional[asyncpg.Pool] = None

    @property
    def database_type(self) -> DatabaseType:
        return DatabaseType.POSTGRESQL

    def _build_dsn(self) -> str:
        """Build connection DSN."""
        user = self.config.user or ""
        password = self.config.password or ""
        auth = f"{user}:{password}@" if user else ""
        port = self.config.port or 5432
        database = self.config.database or "postgres"
        return f"postgresql://{auth}{self.config.host}:{port}/{database}"

    async def connect(self) -> None:
        """Create connection pool."""
        dsn = self._build_dsn()
        self._pool = await asyncpg.create_pool(dsn, **self.config.extra)

    async def disconnect(self) -> None:
        """Close connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None

    async def execute(
        self, query: str, params: Optional[dict[str, Any]] = None
    ) -> QueryResult:
        """Execute SQL query."""
        if not self._pool:
            raise RuntimeError("Not connected")

        async with self._pool.acquire() as conn:
            if params:
                rows = await conn.fetch(query, *params.values())
            else:
                rows = await conn.fetch(query)

            if not rows:
                return QueryResult(columns=[], rows=[], row_count=0)

            columns = list(rows[0].keys())
            data = [list(row.values()) for row in rows]
            return QueryResult(columns=columns, rows=data, row_count=len(data))

    async def health_check(self) -> bool:
        """Check connection health."""
        if not self._pool:
            return False

        try:
            async with self._pool.acquire() as conn:
                result = await conn.fetchval("SELECT 1")
                return result == 1
        except Exception:
            return False
