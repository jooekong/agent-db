"""Qdrant vector database adapter."""

from typing import Any, Optional

from qdrant_client import QdrantClient

from agent_db.adapters.protocol import (
    DatabaseAdapter,
    ConnectionConfig,
    DatabaseType,
    QueryResult,
)
from agent_db.adapters.factory import register_adapter


@register_adapter(DatabaseType.QDRANT)
class QdrantAdapter(DatabaseAdapter):
    """Adapter for Qdrant vector database."""

    def __init__(self, config: ConnectionConfig):
        super().__init__(config)
        self._client: Optional[QdrantClient] = None

    @property
    def database_type(self) -> DatabaseType:
        return DatabaseType.QDRANT

    async def connect(self) -> None:
        """Connect to Qdrant."""
        https = self.config.ssl.enabled
        api_key = self.config.get_password() if self.config.user == "api_key" else None

        self._client = QdrantClient(
            host=self.config.host,
            port=self.config.effective_port,
            https=https,
            api_key=api_key,
            timeout=self.config.pool.connect_timeout,
            **self.config.extra,
        )

    async def disconnect(self) -> None:
        """Disconnect from Qdrant."""
        if self._client:
            self._client.close()
            self._client = None

    async def execute(
        self, query: str, params: Optional[dict[str, Any]] = None
    ) -> QueryResult:
        """Execute is not directly supported for vector DB."""
        raise NotImplementedError("Use search() for Qdrant queries")

    async def search(
        self,
        collection: str,
        vector: list[float],
        limit: int = 10,
        filters: Optional[dict[str, Any]] = None,
    ) -> QueryResult:
        """Search for similar vectors."""
        if not self._client:
            raise RuntimeError("Not connected")

        results = self._client.search(
            collection_name=collection,
            query_vector=vector,
            limit=limit,
            query_filter=filters,
        )

        columns = ["id", "score", "payload"]
        rows = [[str(r.id), r.score, r.payload] for r in results]
        return QueryResult(columns=columns, rows=rows, row_count=len(rows))

    async def health_check(self) -> bool:
        """Check connection health."""
        if not self._client:
            return False

        try:
            self._client.get_collections()
            return True
        except Exception:
            return False
