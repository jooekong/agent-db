"""Nebula Graph database adapter."""

from typing import Any, Optional

from nebula3.gclient.net import ConnectionPool
from nebula3.Config import Config as NebulaConfig

from agent_db.adapters.protocol import (
    DatabaseAdapter,
    ConnectionConfig,
    DatabaseType,
    QueryResult,
)
from agent_db.adapters.factory import register_adapter


@register_adapter(DatabaseType.NEBULA)
class NebulaAdapter(DatabaseAdapter):
    """Adapter for Nebula Graph database."""

    def __init__(self, config: ConnectionConfig):
        super().__init__(config)
        self._pool: Optional[ConnectionPool] = None
        self._space: str = config.database or "default"

    @property
    def database_type(self) -> DatabaseType:
        return DatabaseType.NEBULA

    async def connect(self) -> None:
        """Connect to Nebula Graph."""
        nebula_config = NebulaConfig()
        nebula_config.max_connection_pool_size = self.config.pool.max_size
        nebula_config.timeout = self.config.pool.connect_timeout * 1000  # ms
        nebula_config.idle_time = self.config.pool.max_idle_time * 1000  # ms

        self._pool = ConnectionPool()
        ok = self._pool.init(
            [(self.config.host, self.config.effective_port)],
            nebula_config,
        )
        if not ok:
            raise RuntimeError("Failed to initialize Nebula connection pool")

    async def disconnect(self) -> None:
        """Disconnect from Nebula Graph."""
        if self._pool:
            self._pool.close()
            self._pool = None

    async def execute(
        self, query: str, params: Optional[dict[str, Any]] = None
    ) -> QueryResult:
        """Execute nGQL query."""
        if not self._pool:
            raise RuntimeError("Not connected")

        session = self._pool.get_session(
            self.config.user or "root",
            self.config.get_password() or "nebula",
        )

        try:
            # Switch to space if not already
            if self._space:
                session.execute(f"USE {self._space}")

            # Execute query
            if params:
                # Nebula uses $param syntax for parameters
                for key, value in params.items():
                    if isinstance(value, str):
                        query = query.replace(f"${key}", f'"{value}"')
                    else:
                        query = query.replace(f"${key}", str(value))

            result = session.execute(query)

            if not result.is_succeeded():
                raise RuntimeError(f"Query failed: {result.error_msg()}")

            if result.is_empty():
                return QueryResult(columns=[], rows=[], row_count=0)

            columns = result.keys()
            rows = []
            for row_idx in range(result.row_size()):
                row = []
                for col_idx in range(len(columns)):
                    value = result.row_values(row_idx)[col_idx]
                    row.append(self._convert_value(value))
                rows.append(row)

            return QueryResult(columns=columns, rows=rows, row_count=len(rows))

        finally:
            session.release()

    def _convert_value(self, value: Any) -> Any:
        """Convert Nebula value to Python value."""
        if value.is_null():
            return None
        if value.is_bool():
            return value.as_bool()
        if value.is_int():
            return value.as_int()
        if value.is_double():
            return value.as_double()
        if value.is_string():
            return value.as_string()
        if value.is_list():
            return [self._convert_value(v) for v in value.as_list()]
        if value.is_map():
            return {k: self._convert_value(v) for k, v in value.as_map().items()}
        if value.is_vertex():
            vertex = value.as_node()
            return {"id": vertex.get_id().as_string(), "tags": vertex.tags()}
        if value.is_edge():
            edge = value.as_relationship()
            return {
                "src": edge.start_vertex_id().as_string(),
                "dst": edge.end_vertex_id().as_string(),
                "type": edge.edge_name(),
            }
        if value.is_path():
            path = value.as_path()
            return {
                "nodes": [self._convert_value(n) for n in path.nodes()],
                "relationships": [self._convert_value(r) for r in path.relationships()],
            }
        return str(value)

    async def health_check(self) -> bool:
        """Check connection health."""
        if not self._pool:
            return False

        try:
            session = self._pool.get_session(
                self.config.user or "root",
                self.config.get_password() or "nebula",
            )
            try:
                result = session.execute("SHOW HOSTS")
                return result.is_succeeded()
            finally:
                session.release()
        except Exception:
            return False
