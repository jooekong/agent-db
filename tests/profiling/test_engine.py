"""Tests for profiling engine."""

from datetime import datetime
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_db.adapters.protocol import (
    ConnectionConfig,
    DatabaseAdapter,
    DatabaseType,
    QueryResult,
)
from agent_db.metadata.models import (
    ColumnDataType,
    DataProfile,
    DistributionType,
    ProfilingConfig,
)
from agent_db.profiling.engine import ProfilerEngine
from agent_db.profiling.sampler import PostgreSQLSampler


class MockPostgreSQLAdapter(DatabaseAdapter):
    """Mock PostgreSQL adapter for testing."""

    def __init__(self):
        config = ConnectionConfig(
            database_type=DatabaseType.POSTGRESQL,
            host="localhost",
            database="test_db",
        )
        super().__init__(config)
        self._connected = False

    @property
    def database_type(self) -> DatabaseType:
        return DatabaseType.POSTGRESQL

    async def connect(self) -> None:
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False

    async def execute(
        self, query: str, params: Optional[dict[str, Any]] = None
    ) -> QueryResult:
        """Return mock results based on query type."""
        query_lower = query.lower().strip()

        # Schema hash query (check this first as it also contains column info)
        if "md5" in query_lower and "string_agg" in query_lower:
            return QueryResult(
                columns=["hash"],
                rows=[["abc123def456"]],
                row_count=1,
            )

        # Row count query (simple COUNT(*) without other aggregates)
        # Exclude queries with length(), ::numeric, ::float, or GROUP BY (complex stats queries)
        if ("count(*)" in query_lower and
            "information_schema" not in query_lower and
            "::numeric" not in query_lower and
            "::float" not in query_lower and
            "length(" not in query_lower and
            "group by" not in query_lower):
            return QueryResult(columns=["cnt"], rows=[[1000]], row_count=1)

        # Column info query
        if "information_schema.columns" in query_lower:
            return QueryResult(
                columns=["column_name", "data_type", "is_nullable"],
                rows=[
                    ["id", "integer", "NO"],
                    ["name", "character varying", "YES"],
                    ["amount", "numeric", "YES"],
                    ["created_at", "timestamp", "NO"],
                ],
                row_count=4,
            )

        # Numeric stats query - check ::numeric and NOT multiple percentile_cont (has only 0.5)
        # This matches the main numeric stats query which has AVG(col::numeric)
        if "::numeric" in query_lower and "percentile_cont(0.05)" not in query_lower:
            return QueryResult(
                columns=["min_val", "max_val", "avg_val", "median_val", "distinct_count", "null_ratio"],
                rows=[[1.0, 1000.0, 250.5, 200.0, 500, 0.05]],
                row_count=1,
            )

        # Percentiles query (has multiple percentile_cont calls)
        if "percentile_cont(0.05)" in query_lower:
            return QueryResult(
                columns=["p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99"],
                rows=[[10.0, 25.0, 100.0, 200.0, 400.0, 700.0, 850.0, 950.0]],
                row_count=1,
            )

        # Text stats query (uses LENGTH)
        if "length(" in query_lower:
            return QueryResult(
                columns=["min_length", "max_length", "avg_length", "distinct_count", "empty_ratio", "null_ratio"],
                rows=[[1, 100, 25.5, 800, 0.01, 0.02]],
                row_count=1,
            )

        # Top values query
        if "group by" in query_lower and "order by" in query_lower and "limit" in query_lower:
            return QueryResult(
                columns=["name", "cnt"],
                rows=[["Alice", 50], ["Bob", 45], ["Charlie", 40]],
                row_count=3,
            )

        # Sample column query (single column)
        if "random()" in query_lower and "where" in query_lower and "is not null" in query_lower:
            return QueryResult(
                columns=["value"],
                rows=[[100.0], [200.0], [300.0], [400.0], [500.0]],
                row_count=5,
            )

        # Sample query
        if "random()" in query_lower or "tablesample" in query_lower:
            return QueryResult(
                columns=["id", "name", "amount"],
                rows=[[1, "Alice", 100.0], [2, "Bob", 200.0]],
                row_count=2,
            )

        # Default empty result
        return QueryResult(columns=[], rows=[], row_count=0)

    async def health_check(self) -> bool:
        return self._connected


class TestProfilerEngine:
    """Tests for ProfilerEngine."""

    @pytest.fixture
    def mock_adapter(self):
        """Create mock adapter."""
        return MockPostgreSQLAdapter()

    @pytest.fixture
    def engine(self, mock_adapter):
        """Create profiler engine with mock adapter."""
        config = ProfilingConfig(
            sample_size=100,
            detect_distribution=True,
            calculate_percentiles=True,
            analyze_text=True,
        )
        return ProfilerEngine(mock_adapter, config)

    def test_classify_pg_type_numeric(self, engine):
        """Test numeric type classification."""
        assert engine._classify_pg_type("integer") == ColumnDataType.NUMERIC
        assert engine._classify_pg_type("bigint") == ColumnDataType.NUMERIC
        assert engine._classify_pg_type("numeric") == ColumnDataType.NUMERIC
        assert engine._classify_pg_type("double precision") == ColumnDataType.NUMERIC

    def test_classify_pg_type_text(self, engine):
        """Test text type classification."""
        assert engine._classify_pg_type("character varying") == ColumnDataType.TEXT
        assert engine._classify_pg_type("varchar") == ColumnDataType.TEXT
        assert engine._classify_pg_type("text") == ColumnDataType.TEXT

    def test_classify_pg_type_timestamp(self, engine):
        """Test timestamp type classification."""
        assert engine._classify_pg_type("timestamp") == ColumnDataType.TIMESTAMP
        assert engine._classify_pg_type("timestamptz") == ColumnDataType.TIMESTAMP
        assert engine._classify_pg_type("date") == ColumnDataType.TIMESTAMP

    def test_classify_pg_type_boolean(self, engine):
        """Test boolean type classification."""
        assert engine._classify_pg_type("boolean") == ColumnDataType.BOOLEAN

    def test_classify_pg_type_unknown(self, engine):
        """Test unknown type classification."""
        assert engine._classify_pg_type("custom_type") == ColumnDataType.UNKNOWN

    @pytest.mark.asyncio
    async def test_profile_table(self, engine):
        """Test full table profiling."""
        profile = await engine.profile_table("users", "test_db")

        assert profile.table == "users"
        assert profile.database == "test_db"
        assert profile.row_count == 1000
        assert len(profile.columns) == 4
        assert profile.schema_hash == "abc123def456"

    @pytest.mark.asyncio
    async def test_profile_numeric_column(self, engine):
        """Test numeric column profiling."""
        profile = await engine.profile_table("users", "test_db")

        # Find numeric column (id or amount)
        numeric_col = next(
            (c for c in profile.columns if c.data_type == ColumnDataType.NUMERIC),
            None,
        )

        assert numeric_col is not None
        assert numeric_col.min_val is not None
        assert numeric_col.max_val is not None
        assert numeric_col.avg_val is not None
        assert numeric_col.percentiles is not None

    @pytest.mark.asyncio
    async def test_profile_text_column(self, engine):
        """Test text column profiling."""
        profile = await engine.profile_table("users", "test_db")

        # Find text column (name)
        text_col = next(
            (c for c in profile.columns if c.data_type == ColumnDataType.TEXT),
            None,
        )

        assert text_col is not None
        assert text_col.text_stats is not None
        assert text_col.text_stats.min_length >= 0
        assert text_col.text_stats.max_length > 0

    @pytest.mark.asyncio
    async def test_should_update_profile_no_existing(self, engine):
        """Test that new tables always need profiling."""
        result = await engine.should_update_profile("new_table", None)

        assert result is True

    @pytest.mark.asyncio
    async def test_should_update_profile_schema_changed(self, engine):
        """Test that schema changes trigger update."""
        existing = DataProfile(
            table="users",
            database="test_db",
            last_updated=datetime.utcnow(),
            row_count=1000,
            columns=[],
            schema_hash="old_hash_different",
        )

        result = await engine.should_update_profile("users", existing)

        assert result is True

    @pytest.mark.asyncio
    async def test_should_update_profile_no_change(self, engine):
        """Test that unchanged schema doesn't trigger update."""
        existing = DataProfile(
            table="users",
            database="test_db",
            last_updated=datetime.utcnow(),
            row_count=1000,
            columns=[],
            schema_hash="abc123def456",  # Matches mock response
        )

        result = await engine.should_update_profile("users", existing)

        assert result is False

    def test_compute_schema_hash(self):
        """Test schema hash computation."""
        column_info = [
            {"name": "id", "data_type": "integer"},
            {"name": "name", "data_type": "varchar"},
        ]

        hash1 = ProfilerEngine.compute_schema_hash(column_info)
        hash2 = ProfilerEngine.compute_schema_hash(column_info)

        assert hash1 == hash2
        assert len(hash1) == 32  # MD5 hex length

    def test_compute_schema_hash_different(self):
        """Test that different schemas produce different hashes."""
        info1 = [{"name": "id", "data_type": "integer"}]
        info2 = [{"name": "id", "data_type": "bigint"}]

        hash1 = ProfilerEngine.compute_schema_hash(info1)
        hash2 = ProfilerEngine.compute_schema_hash(info2)

        assert hash1 != hash2
