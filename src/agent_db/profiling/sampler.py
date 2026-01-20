"""Sampling strategies for different databases."""

from abc import ABC, abstractmethod
from typing import Any, Optional

from agent_db.adapters.protocol import DatabaseAdapter, DatabaseType, QueryResult
from agent_db.profiling.sql_templates import SQLTemplates


class BaseSampler(ABC):
    """Base class for database samplers."""

    def __init__(self, adapter: DatabaseAdapter):
        self.adapter = adapter

    @abstractmethod
    async def sample_table(
        self, table: str, sample_size: int = 10000
    ) -> QueryResult:
        """Sample rows from table."""
        pass

    @abstractmethod
    async def sample_column(
        self, table: str, column: str, sample_size: int = 10000
    ) -> list[Any]:
        """Sample values from a specific column."""
        pass

    @abstractmethod
    async def get_row_count(self, table: str) -> int:
        """Get total row count."""
        pass

    @abstractmethod
    async def get_column_info(self, table: str) -> list[dict[str, Any]]:
        """Get column metadata (name, type, nullable)."""
        pass


class PostgreSQLSampler(BaseSampler):
    """Sampler for PostgreSQL databases."""

    async def get_row_count(self, table: str) -> int:
        """Get total row count."""
        query = SQLTemplates.render("postgresql", "row_count", table=table)
        result = await self.adapter.execute(query)
        if result.rows:
            return int(result.rows[0][0])
        return 0

    async def get_column_info(self, table: str) -> list[dict[str, Any]]:
        """Get column metadata."""
        query = SQLTemplates.render("postgresql", "column_info", table=table)
        result = await self.adapter.execute(query)
        columns = []
        for row in result.rows:
            columns.append({
                "name": row[0],
                "data_type": row[1],
                "nullable": row[2] == "YES",
            })
        return columns

    async def sample_table(
        self, table: str, sample_size: int = 10000
    ) -> QueryResult:
        """Sample rows using TABLESAMPLE or ORDER BY RANDOM()."""
        row_count = await self.get_row_count(table)

        if row_count == 0:
            return QueryResult(columns=[], rows=[], row_count=0)

        # Use TABLESAMPLE for large tables
        if row_count > 100000:
            percentage = min(100, (sample_size / row_count) * 100 * 1.2)
            query = SQLTemplates.render(
                "postgresql", "sample_tablesample",
                table=table, percentage=percentage
            )
        else:
            query = SQLTemplates.render(
                "postgresql", "sample_random",
                table=table, sample_size=sample_size
            )

        return await self.adapter.execute(query)

    async def sample_column(
        self, table: str, column: str, sample_size: int = 10000
    ) -> list[Any]:
        """Sample values from a column."""
        query = SQLTemplates.render(
            "postgresql", "sample_column",
            table=table, column=column, sample_size=sample_size
        )
        result = await self.adapter.execute(query)
        return [row[0] for row in result.rows]

    async def get_numeric_stats(
        self, table: str, column: str
    ) -> dict[str, Any]:
        """Get numeric column statistics."""
        query = SQLTemplates.render(
            "postgresql", "numeric_stats",
            table=table, column=column
        )
        result = await self.adapter.execute(query)
        if result.rows:
            row = result.rows[0]
            return {
                "min_val": float(row[0]) if row[0] is not None else None,
                "max_val": float(row[1]) if row[1] is not None else None,
                "avg_val": float(row[2]) if row[2] is not None else None,
                "median_val": float(row[3]) if row[3] is not None else None,
                "distinct_count": int(row[4]) if row[4] is not None else None,
                "null_ratio": float(row[5]) if row[5] is not None else None,
            }
        return {}

    async def get_numeric_percentiles(
        self, table: str, column: str
    ) -> dict[str, float]:
        """Get numeric column percentiles."""
        query = SQLTemplates.render(
            "postgresql", "numeric_percentiles",
            table=table, column=column
        )
        result = await self.adapter.execute(query)
        if result.rows:
            row = result.rows[0]
            return {
                "p5": float(row[0]) if row[0] is not None else None,
                "p10": float(row[1]) if row[1] is not None else None,
                "p25": float(row[2]) if row[2] is not None else None,
                "p50": float(row[3]) if row[3] is not None else None,
                "p75": float(row[4]) if row[4] is not None else None,
                "p90": float(row[5]) if row[5] is not None else None,
                "p95": float(row[6]) if row[6] is not None else None,
                "p99": float(row[7]) if row[7] is not None else None,
            }
        return {}

    async def get_text_stats(
        self, table: str, column: str
    ) -> dict[str, Any]:
        """Get text column statistics."""
        query = SQLTemplates.render(
            "postgresql", "text_stats",
            table=table, column=column
        )
        result = await self.adapter.execute(query)
        if result.rows:
            row = result.rows[0]
            return {
                "min_length": int(row[0]) if row[0] is not None else 0,
                "max_length": int(row[1]) if row[1] is not None else 0,
                "avg_length": float(row[2]) if row[2] is not None else 0.0,
                "distinct_count": int(row[3]) if row[3] is not None else None,
                "empty_ratio": float(row[4]) if row[4] is not None else 0.0,
                "null_ratio": float(row[5]) if row[5] is not None else 0.0,
            }
        return {}

    async def get_top_values(
        self, table: str, column: str, limit: int = 10
    ) -> list[tuple[str, int]]:
        """Get most frequent values."""
        query = SQLTemplates.render(
            "postgresql", "top_values",
            table=table, column=column, limit=limit
        )
        result = await self.adapter.execute(query)
        return [(str(row[0]), int(row[1])) for row in result.rows]

    async def get_schema_hash(self, table: str) -> str:
        """Get hash of table schema for change detection."""
        query = SQLTemplates.render("postgresql", "schema_hash", table=table)
        result = await self.adapter.execute(query)
        if result.rows and result.rows[0][0]:
            return str(result.rows[0][0])
        return ""


class Neo4jSampler(BaseSampler):
    """Sampler for Neo4j graph databases."""

    async def get_row_count(self, table: str) -> int:
        """Get node count for label."""
        query = SQLTemplates.render("neo4j", "row_count", label=table)
        result = await self.adapter.execute(query)
        if result.rows:
            return int(result.rows[0][0])
        return 0

    async def get_column_info(self, table: str) -> list[dict[str, Any]]:
        """Get property keys for label."""
        query = SQLTemplates.render("neo4j", "property_keys", label=table)
        result = await self.adapter.execute(query)
        return [{"name": row[0], "data_type": "unknown", "nullable": True}
                for row in result.rows]

    async def sample_table(
        self, table: str, sample_size: int = 10000
    ) -> QueryResult:
        """Sample nodes."""
        query = SQLTemplates.render(
            "neo4j", "sample_random",
            label=table, sample_size=sample_size
        )
        return await self.adapter.execute(query)

    async def sample_column(
        self, table: str, column: str, sample_size: int = 10000
    ) -> list[Any]:
        """Sample property values."""
        query = SQLTemplates.render(
            "neo4j", "sample_property",
            label=table, property=column, sample_size=sample_size
        )
        result = await self.adapter.execute(query)
        return [row[0] for row in result.rows]

    async def get_numeric_stats(
        self, table: str, column: str
    ) -> dict[str, Any]:
        """Get numeric property statistics."""
        query = SQLTemplates.render(
            "neo4j", "numeric_stats",
            label=table, property=column
        )
        result = await self.adapter.execute(query)
        if result.rows:
            row = result.rows[0]
            return {
                "min_val": float(row[0]) if row[0] is not None else None,
                "max_val": float(row[1]) if row[1] is not None else None,
                "avg_val": float(row[2]) if row[2] is not None else None,
                "distinct_count": int(row[3]) if row[3] is not None else None,
            }
        return {}

    async def get_null_ratio(self, table: str, column: str) -> float:
        """Get null ratio for property."""
        query = SQLTemplates.render(
            "neo4j", "null_ratio",
            label=table, property=column
        )
        result = await self.adapter.execute(query)
        if result.rows and result.rows[0][0] is not None:
            return float(result.rows[0][0])
        return 0.0


class InfluxDBSampler(BaseSampler):
    """Sampler for InfluxDB time-series databases."""

    def __init__(
        self,
        adapter: DatabaseAdapter,
        bucket: str,
        start: str = "-30d",
        stop: str = "now()",
    ):
        super().__init__(adapter)
        self.bucket = bucket
        self.start = start
        self.stop = stop

    async def get_row_count(self, table: str) -> int:
        """Get point count for measurement."""
        query = SQLTemplates.render(
            "influxdb", "row_count",
            bucket=self.bucket, measurement=table,
            start=self.start, stop=self.stop
        )
        result = await self.adapter.execute(query)
        if result.rows:
            return int(result.rows[0][0])
        return 0

    async def get_column_info(self, table: str) -> list[dict[str, Any]]:
        """Get field keys for measurement."""
        query = SQLTemplates.render(
            "influxdb", "field_keys",
            bucket=self.bucket, measurement=table
        )
        result = await self.adapter.execute(query)
        return [{"name": row[0], "data_type": "numeric", "nullable": True}
                for row in result.rows]

    async def sample_table(
        self, table: str, sample_size: int = 10000
    ) -> QueryResult:
        """Sample points from measurement."""
        # InfluxDB doesn't have a generic table sample
        # We'd need to sample each field separately
        return QueryResult(columns=[], rows=[], row_count=0)

    async def sample_column(
        self, table: str, column: str, sample_size: int = 10000
    ) -> list[Any]:
        """Sample field values."""
        query = SQLTemplates.render(
            "influxdb", "sample_random",
            bucket=self.bucket, measurement=table,
            field=column, sample_size=sample_size,
            start=self.start, stop=self.stop
        )
        result = await self.adapter.execute(query)
        return [row[0] for row in result.rows]

    async def get_percentile(
        self, table: str, column: str, quantile: float
    ) -> Optional[float]:
        """Get specific percentile."""
        query = SQLTemplates.render(
            "influxdb", "percentiles",
            bucket=self.bucket, measurement=table,
            field=column, quantile=quantile,
            start=self.start, stop=self.stop
        )
        result = await self.adapter.execute(query)
        if result.rows:
            return float(result.rows[0][0])
        return None


class MySQLSampler(BaseSampler):
    """Sampler for MySQL databases."""

    async def get_row_count(self, table: str) -> int:
        """Get total row count."""
        result = await self.adapter.execute(f"SELECT COUNT(*) FROM `{table}`")
        return int(result.rows[0][0]) if result.rows else 0

    async def get_column_info(self, table: str) -> list[dict[str, Any]]:
        """Get column metadata."""
        query = f"""
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_schema = DATABASE() AND table_name = '{table}'
        ORDER BY ordinal_position
        """
        result = await self.adapter.execute(query)
        columns = []
        for row in result.rows:
            columns.append({
                "name": row[0],
                "data_type": row[1],
                "nullable": row[2] == "YES",
            })
        return columns

    async def sample_table(
        self, table: str, sample_size: int = 10000
    ) -> QueryResult:
        """Sample rows using ORDER BY RAND()."""
        row_count = await self.get_row_count(table)

        if row_count == 0:
            return QueryResult(columns=[], rows=[], row_count=0)

        query = f"SELECT * FROM `{table}` ORDER BY RAND() LIMIT {sample_size}"
        return await self.adapter.execute(query)

    async def sample_column(
        self, table: str, column: str, sample_size: int = 10000
    ) -> list[Any]:
        """Sample values from a specific column."""
        query = f"SELECT `{column}` FROM `{table}` ORDER BY RAND() LIMIT {sample_size}"
        result = await self.adapter.execute(query)
        return [row[0] for row in result.rows]


def get_sampler(adapter: DatabaseAdapter, **kwargs: Any) -> BaseSampler:
    """Factory function to get appropriate sampler for adapter."""
    samplers = {
        DatabaseType.POSTGRESQL: PostgreSQLSampler,
        DatabaseType.MYSQL: MySQLSampler,
        DatabaseType.NEO4J: Neo4jSampler,
        DatabaseType.INFLUXDB: InfluxDBSampler,
    }

    sampler_class = samplers.get(adapter.database_type)
    if not sampler_class:
        raise ValueError(f"No sampler for database type: {adapter.database_type}")

    if adapter.database_type == DatabaseType.INFLUXDB:
        bucket = kwargs.get("bucket", "default")
        return InfluxDBSampler(adapter, bucket=bucket, **kwargs)

    return sampler_class(adapter)
