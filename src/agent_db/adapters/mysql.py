"""MySQL database adapter."""

from typing import Any, Optional

import aiomysql

from agent_db.adapters.protocol import (
    DatabaseAdapter,
    ConnectionConfig,
    DatabaseType,
    QueryResult,
)
from agent_db.adapters.factory import register_adapter


@register_adapter(DatabaseType.MYSQL)
class MySQLAdapter(DatabaseAdapter):
    """Adapter for MySQL databases."""

    def __init__(self, config: ConnectionConfig):
        super().__init__(config)
        self._pool: Optional[aiomysql.Pool] = None

    @property
    def database_type(self) -> DatabaseType:
        return DatabaseType.MYSQL

    def _build_ssl_context(self) -> Optional[dict[str, Any]]:
        """Build SSL context dict if configured."""
        if not self.config.ssl.enabled:
            return None
        ssl_ctx: dict[str, Any] = {}
        if self.config.ssl.ca_cert:
            ssl_ctx["ca"] = self.config.ssl.ca_cert
        if self.config.ssl.client_cert:
            ssl_ctx["cert"] = self.config.ssl.client_cert
        if self.config.ssl.client_key:
            ssl_ctx["key"] = self.config.ssl.client_key
        ssl_ctx["check_hostname"] = self.config.ssl.verify
        return ssl_ctx

    async def connect(self) -> None:
        """Create connection pool."""
        ssl_ctx = self._build_ssl_context()
        pool_config = self.config.pool

        self._pool = await aiomysql.create_pool(
            host=self.config.host,
            port=self.config.effective_port,
            user=self.config.user or "root",
            password=self.config.get_password() or "",
            db=self.config.database or "mysql",
            minsize=pool_config.min_size,
            maxsize=pool_config.max_size,
            connect_timeout=pool_config.connect_timeout,
            ssl=ssl_ctx,
            autocommit=True,
            **self.config.extra,
        )

    async def disconnect(self) -> None:
        """Close connection pool."""
        if self._pool:
            self._pool.close()
            await self._pool.wait_closed()
            self._pool = None

    async def execute(
        self, query: str, params: Optional[dict[str, Any]] = None
    ) -> QueryResult:
        """Execute SQL query."""
        if not self._pool:
            raise RuntimeError("Not connected")

        async with self._pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                if params:
                    await cursor.execute(query, tuple(params.values()))
                else:
                    await cursor.execute(query)

                rows = await cursor.fetchall()

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
                async with conn.cursor() as cursor:
                    await cursor.execute("SELECT 1")
                    result = await cursor.fetchone()
                    return result is not None and result[0] == 1
        except Exception:
            return False
