"""Neo4j graph database adapter."""

from typing import Any, Optional

from neo4j import GraphDatabase

from agent_db.adapters.protocol import (
    DatabaseAdapter,
    ConnectionConfig,
    DatabaseType,
    QueryResult,
)


class Neo4jAdapter(DatabaseAdapter):
    """Adapter for Neo4j graph database."""

    def __init__(self, config: ConnectionConfig):
        super().__init__(config)
        self._driver = None

    @property
    def database_type(self) -> DatabaseType:
        return DatabaseType.NEO4J

    def _build_uri(self) -> str:
        """Build connection URI."""
        port = self.config.port or 7687
        return f"bolt://{self.config.host}:{port}"

    async def connect(self) -> None:
        """Connect to Neo4j."""
        uri = self._build_uri()
        auth = None
        if self.config.user and self.config.password:
            auth = (self.config.user, self.config.password)
        self._driver = GraphDatabase.driver(uri, auth=auth, **self.config.extra)

    async def disconnect(self) -> None:
        """Disconnect from Neo4j."""
        if self._driver:
            self._driver.close()
            self._driver = None

    async def execute(
        self, query: str, params: Optional[dict[str, Any]] = None
    ) -> QueryResult:
        """Execute Cypher query."""
        if not self._driver:
            raise RuntimeError("Not connected")

        with self._driver.session() as session:
            result = session.run(query, params or {})
            data = result.data()

            if not data:
                return QueryResult(columns=[], rows=[], row_count=0)

            columns = list(data[0].keys())
            rows = [list(record.values()) for record in data]
            return QueryResult(columns=columns, rows=rows, row_count=len(rows))

    async def health_check(self) -> bool:
        """Check connection health."""
        if not self._driver:
            return False

        try:
            with self._driver.session() as session:
                session.run("RETURN 1")
                return True
        except Exception:
            return False
