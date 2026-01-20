"""Milvus vector database adapter."""

from typing import Any, Optional

from pymilvus import MilvusClient, DataType

from agent_db.adapters.protocol import (
    DatabaseAdapter,
    ConnectionConfig,
    DatabaseType,
    QueryResult,
)
from agent_db.adapters.factory import register_adapter


@register_adapter(DatabaseType.MILVUS)
class MilvusAdapter(DatabaseAdapter):
    """Adapter for Milvus vector database."""

    def __init__(self, config: ConnectionConfig):
        super().__init__(config)
        self._client: Optional[MilvusClient] = None

    @property
    def database_type(self) -> DatabaseType:
        return DatabaseType.MILVUS

    def _build_uri(self) -> str:
        """Build connection URI."""
        scheme = "https" if self.config.ssl.enabled else "http"
        return f"{scheme}://{self.config.host}:{self.config.effective_port}"

    async def connect(self) -> None:
        """Connect to Milvus."""
        uri = self._build_uri()
        token = None
        if self.config.user and self.config.get_password():
            token = f"{self.config.user}:{self.config.get_password()}"

        self._client = MilvusClient(
            uri=uri,
            token=token,
            db_name=self.config.database or "default",
            timeout=self.config.pool.connect_timeout,
            **self.config.extra,
        )

    async def disconnect(self) -> None:
        """Disconnect from Milvus."""
        if self._client:
            self._client.close()
            self._client = None

    async def execute(
        self, query: str, params: Optional[dict[str, Any]] = None
    ) -> QueryResult:
        """Execute is not directly supported for vector DB."""
        raise NotImplementedError("Use search() or query() for Milvus operations")

    async def search(
        self,
        collection: str,
        vector: list[float],
        limit: int = 10,
        output_fields: Optional[list[str]] = None,
        filters: Optional[str] = None,
    ) -> QueryResult:
        """Search for similar vectors."""
        if not self._client:
            raise RuntimeError("Not connected")

        results = self._client.search(
            collection_name=collection,
            data=[vector],
            limit=limit,
            output_fields=output_fields or ["*"],
            filter=filters,
        )

        if not results or not results[0]:
            return QueryResult(columns=["id", "distance"], rows=[], row_count=0)

        columns = ["id", "distance"]
        if output_fields:
            columns.extend(output_fields)

        rows = []
        for hit in results[0]:
            row = [hit["id"], hit["distance"]]
            entity = hit.get("entity", {})
            if output_fields:
                for field in output_fields:
                    row.append(entity.get(field))
            rows.append(row)

        return QueryResult(columns=columns, rows=rows, row_count=len(rows))

    async def query(
        self,
        collection: str,
        filters: str,
        output_fields: Optional[list[str]] = None,
        limit: int = 100,
    ) -> QueryResult:
        """Query by filter expression."""
        if not self._client:
            raise RuntimeError("Not connected")

        results = self._client.query(
            collection_name=collection,
            filter=filters,
            output_fields=output_fields or ["*"],
            limit=limit,
        )

        if not results:
            return QueryResult(columns=[], rows=[], row_count=0)

        columns = list(results[0].keys())
        rows = [list(row.values()) for row in results]
        return QueryResult(columns=columns, rows=rows, row_count=len(rows))

    async def health_check(self) -> bool:
        """Check connection health."""
        if not self._client:
            return False

        try:
            self._client.list_collections()
            return True
        except Exception:
            return False
