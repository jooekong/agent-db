"""InfluxDB time series database adapter."""

from typing import Any, Optional

from influxdb_client import InfluxDBClient

from agent_db.adapters.protocol import (
    DatabaseAdapter,
    ConnectionConfig,
    DatabaseType,
    QueryResult,
)
from agent_db.adapters.factory import register_adapter


@register_adapter(DatabaseType.INFLUXDB)
class InfluxDBAdapter(DatabaseAdapter):
    """Adapter for InfluxDB time series database."""

    def __init__(self, config: ConnectionConfig):
        super().__init__(config)
        self._client: Optional[InfluxDBClient] = None

    @property
    def database_type(self) -> DatabaseType:
        return DatabaseType.INFLUXDB

    async def connect(self) -> None:
        """Connect to InfluxDB."""
        scheme = "https" if self.config.ssl.enabled else "http"
        url = f"{scheme}://{self.config.host}:{self.config.effective_port}"
        token = self.config.get_password() or self.config.extra.get("token", "")
        org = self.config.extra.get("org", "")

        ssl_ca_cert = self.config.ssl.ca_cert if self.config.ssl.enabled else None
        verify_ssl = self.config.ssl.verify if self.config.ssl.enabled else True

        self._client = InfluxDBClient(
            url=url,
            token=token,
            org=org,
            timeout=self.config.pool.connect_timeout * 1000,  # ms
            ssl_ca_cert=ssl_ca_cert,
            verify_ssl=verify_ssl,
        )

    async def disconnect(self) -> None:
        """Disconnect from InfluxDB."""
        if self._client:
            self._client.close()
            self._client = None

    async def execute(
        self, query: str, params: Optional[dict[str, Any]] = None
    ) -> QueryResult:
        """Execute Flux query."""
        if not self._client:
            raise RuntimeError("Not connected")

        query_api = self._client.query_api()
        tables = query_api.query(query)

        all_rows = []
        columns = []

        for table in tables:
            for record in table.records:
                if not columns:
                    columns = list(record.values.keys())
                all_rows.append(list(record.values.values()))

        return QueryResult(
            columns=columns,
            rows=all_rows,
            row_count=len(all_rows),
        )

    async def health_check(self) -> bool:
        """Check connection health."""
        if not self._client:
            return False

        try:
            return self._client.ping()
        except Exception:
            return False
