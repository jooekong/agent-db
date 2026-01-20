"""PostgreSQL database adapter."""

from typing import Any, Optional

import asyncpg

from agent_db.adapters.protocol import (
    DatabaseAdapter,
    ConnectionConfig,
    DatabaseType,
    QueryResult,
)
from agent_db.adapters.factory import register_adapter


@register_adapter(DatabaseType.POSTGRESQL)
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
        password = self.config.get_password() or ""
        auth = f"{user}:{password}@" if user else ""
        database = self.config.database or "postgres"
        return f"postgresql://{auth}{self.config.host}:{self.config.effective_port}/{database}"

    def _build_ssl_context(self) -> Optional[Any]:
        """Build SSL context if configured."""
        if not self.config.ssl.enabled:
            return None
        import ssl
        ctx = ssl.create_default_context()
        if self.config.ssl.ca_cert:
            ctx.load_verify_locations(self.config.ssl.ca_cert)
        if self.config.ssl.client_cert and self.config.ssl.client_key:
            ctx.load_cert_chain(self.config.ssl.client_cert, self.config.ssl.client_key)
        if not self.config.ssl.verify:
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
        return ctx

    async def connect(self) -> None:
        """Create connection pool."""
        dsn = self._build_dsn()
        ssl_ctx = self._build_ssl_context()
        pool_config = self.config.pool

        self._pool = await asyncpg.create_pool(
            dsn,
            min_size=pool_config.min_size,
            max_size=pool_config.max_size,
            max_inactive_connection_lifetime=pool_config.max_idle_time,
            timeout=pool_config.connect_timeout,
            command_timeout=pool_config.command_timeout,
            ssl=ssl_ctx,
            **self.config.extra,
        )

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
