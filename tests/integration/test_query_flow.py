"""Integration tests for the complete query flow."""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_db.adapters.protocol import ConnectionConfig, DatabaseType, QueryResult
from agent_db.adapters.postgresql import PostgreSQLAdapter
from agent_db.engine.planner import QueryPlanner, QueryPlan, QueryStep, StepType
from agent_db.engine.executor import QueryExecutor, ExecutionResult
from agent_db.engine.interpreter import ResultInterpreter
from agent_db.identity.store import MappingStore
from agent_db.identity.models import IdentityLink
from agent_db.semantic.loader import SchemaLoader


# Sample schema for testing
SAMPLE_SCHEMA = """
entities:
  - name: user
    table: users
    description: "Application users"
    semantic_type: actor
    attributes:
      - column: email
        semantic_type: identifier
      - column: name
        semantic_type: dimension
      - column: status
        semantic_type: dimension
        enum_values:
          - value: active
            meaning: "Active user"
          - value: inactive
            meaning: "Inactive user"
    identity:
      canonical_id: user_id
      sources:
        - database: postgresql
          entity: user
          key_column: id
          field_map:
            email: email
            name: name
        - database: qdrant
          collection: user_vectors
          key_column: user_id
          field_map:
            email: user_email

  - name: order
    table: orders
    description: "Customer orders"
    semantic_type: event
    attributes:
      - column: user_id
        semantic_type: identifier
      - column: amount
        semantic_type: measure
      - column: status
        semantic_type: dimension

cross_database_mappings:
  - name: user_unified_view
    sources:
      - database: postgresql
        entity: user
        role: master
      - database: qdrant
        collection: user_vectors
        role: enrichment
"""


@pytest.fixture
def schema():
    """Load test schema."""
    loader = SchemaLoader()
    return loader.load_from_string(SAMPLE_SCHEMA)


@pytest.fixture
def temp_identity_path():
    """Create temporary path for identity store."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "identity"


@pytest.fixture
def mapping_store(temp_identity_path):
    """Create mapping store with pre-populated links."""
    store = MappingStore(temp_identity_path)
    
    # Pre-populate some identity mappings
    links = [
        IdentityLink(
            canonical_id="postgresql:1",
            source="postgresql",
            source_key="1",
            match_rule="primary",
            confidence=1.0,
        ),
        IdentityLink(
            canonical_id="postgresql:1",
            source="qdrant",
            source_key="u_vec_1",
            match_rule="email_match",
            confidence=0.95,
        ),
        IdentityLink(
            canonical_id="postgresql:2",
            source="postgresql",
            source_key="2",
            match_rule="primary",
            confidence=1.0,
        ),
        IdentityLink(
            canonical_id="postgresql:2",
            source="qdrant",
            source_key="u_vec_2",
            match_rule="email_match",
            confidence=0.95,
        ),
    ]
    for link in links:
        store.save_link(link)
    
    return store


class TestQueryPlanExecution:
    """Test query plan execution flow."""

    @pytest.fixture
    def mock_pg_adapter(self):
        """Create mock PostgreSQL adapter."""
        adapter = MagicMock(spec=PostgreSQLAdapter)
        adapter.database_type = DatabaseType.POSTGRESQL
        return adapter

    @pytest.mark.asyncio
    async def test_simple_query_execution(self, schema, mapping_store, mock_pg_adapter):
        """Test executing a simple single-step query."""
        # Setup mock response
        mock_pg_adapter.execute = AsyncMock(return_value=QueryResult(
            columns=["id", "name", "email", "status"],
            rows=[
                [1, "Alice", "alice@example.com", "active"],
                [2, "Bob", "bob@example.com", "active"],
                [3, "Charlie", "charlie@example.com", "inactive"],
            ],
            row_count=3,
        ))

        # Create executor
        executor = QueryExecutor(
            adapters={"postgresql": mock_pg_adapter},
            schema=schema,
            mapping_store=mapping_store,
        )

        # Create a simple query plan
        plan = QueryPlan(
            steps=[
                QueryStep(
                    step_id=1,
                    description="Get all active users",
                    database="postgresql",
                    query="SELECT id, name, email, status FROM users WHERE status = 'active'",
                    depends_on=[],
                )
            ],
            reasoning="Simple query to fetch active users",
        )

        # Execute
        result = await executor.execute(plan)

        # Verify
        assert result.success is True
        assert 1 in result.step_results
        assert result.step_results[1].row_count == 3
        mock_pg_adapter.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_multi_step_query_with_dependency(self, schema, mapping_store, mock_pg_adapter):
        """Test executing multi-step query with dependencies."""
        # Setup mock responses for different queries
        call_count = [0]
        
        async def mock_execute(query, params=None):
            call_count[0] += 1
            if "COUNT" in query:
                return QueryResult(
                    columns=["count"],
                    rows=[[100]],
                    row_count=1,
                )
            elif "users" in query.lower():
                return QueryResult(
                    columns=["id", "name"],
                    rows=[[1, "Alice"], [2, "Bob"]],
                    row_count=2,
                )
            else:
                return QueryResult(
                    columns=["user_id", "total_orders", "total_amount"],
                    rows=[[1, 5, 500.0], [2, 3, 300.0]],
                    row_count=2,
                )

        mock_pg_adapter.execute = AsyncMock(side_effect=mock_execute)

        executor = QueryExecutor(
            adapters={"postgresql": mock_pg_adapter},
            schema=schema,
            mapping_store=mapping_store,
        )

        # Multi-step plan: get users, then get their order stats
        plan = QueryPlan(
            steps=[
                QueryStep(
                    step_id=1,
                    description="Get active users",
                    database="postgresql",
                    query="SELECT id, name FROM users WHERE status = 'active'",
                    depends_on=[],
                ),
                QueryStep(
                    step_id=2,
                    description="Get order statistics for users",
                    database="postgresql",
                    query="SELECT user_id, COUNT(*) as total_orders, SUM(amount) as total_amount FROM orders GROUP BY user_id",
                    depends_on=[1],
                ),
            ],
            reasoning="Get users and their order statistics",
        )

        result = await executor.execute(plan)

        assert result.success is True
        assert len(result.step_results) == 2
        assert call_count[0] == 2

    @pytest.mark.asyncio
    async def test_identity_resolution_step(self, schema, mapping_store, mock_pg_adapter):
        """Test query plan with identity resolution step."""
        mock_pg_adapter.execute = AsyncMock(return_value=QueryResult(
            columns=["id", "name", "email"],
            rows=[
                [1, "Alice", "alice@example.com"],
                [2, "Bob", "bob@example.com"],
            ],
            row_count=2,
        ))

        executor = QueryExecutor(
            adapters={"postgresql": mock_pg_adapter},
            schema=schema,
            mapping_store=mapping_store,
        )

        # Plan with identity resolution
        plan = QueryPlan(
            steps=[
                QueryStep(
                    step_id=1,
                    description="Get users from PostgreSQL",
                    database="postgresql",
                    query="SELECT id, name, email FROM users",
                    depends_on=[],
                ),
                QueryStep(
                    step_id=2,
                    step_type=StepType.RESOLVE_IDENTITY,
                    description="Resolve user identities",
                    database="postgresql",
                    mapping_name="user",  # Entity name, not cross_database_mapping name
                    input_from=1,
                    input_key="id",
                    depends_on=[1],
                ),
            ],
            explanation="Get users and resolve their cross-database identities",
        )

        result = await executor.execute(plan)

        assert result.success is True
        assert 1 in result.step_results
        assert 2 in result.identity_results
        
        # Check identity resolution results
        identity_result = result.identity_results[2]
        assert len(identity_result.canonical_ids) == 2
        assert "postgresql:1" in identity_result.canonical_ids
        assert "postgresql:2" in identity_result.canonical_ids

    @pytest.mark.asyncio
    async def test_parallel_query_execution(self, schema, mapping_store, mock_pg_adapter):
        """Test that independent queries execute in parallel."""
        execution_times = []
        
        async def slow_execute(query, params=None):
            start = asyncio.get_event_loop().time()
            await asyncio.sleep(0.1)  # Simulate slow query
            end = asyncio.get_event_loop().time()
            execution_times.append((start, end))
            return QueryResult(columns=["result"], rows=[[1]], row_count=1)

        mock_pg_adapter.execute = AsyncMock(side_effect=slow_execute)

        executor = QueryExecutor(
            adapters={"postgresql": mock_pg_adapter},
            schema=schema,
            mapping_store=mapping_store,
        )

        # Three independent queries (no dependencies between them)
        plan = QueryPlan(
            steps=[
                QueryStep(step_id=1, description="Query 1", database="postgresql", 
                         query="SELECT 1", depends_on=[]),
                QueryStep(step_id=2, description="Query 2", database="postgresql",
                         query="SELECT 2", depends_on=[]),
                QueryStep(step_id=3, description="Query 3", database="postgresql",
                         query="SELECT 3", depends_on=[]),
            ],
            reasoning="Three independent queries",
        )

        start_time = asyncio.get_event_loop().time()
        result = await executor.execute(plan)
        total_time = asyncio.get_event_loop().time() - start_time

        assert result.success is True
        assert len(result.step_results) == 3
        
        # If parallel: ~0.1s, if sequential: ~0.3s
        # Allow some margin for test flakiness
        assert total_time < 0.25, f"Queries should run in parallel, took {total_time}s"

    @pytest.mark.asyncio
    async def test_failed_step_marks_result_failed(self, schema, mapping_store, mock_pg_adapter):
        """Test that a failed step marks the overall result as failed."""
        async def failing_execute(query, params=None):
            if "fail" in query.lower():
                raise RuntimeError("Database connection lost")
            return QueryResult(columns=["id"], rows=[[1]], row_count=1)

        mock_pg_adapter.execute = AsyncMock(side_effect=failing_execute)

        executor = QueryExecutor(
            adapters={"postgresql": mock_pg_adapter},
            schema=schema,
            mapping_store=mapping_store,
        )

        plan = QueryPlan(
            steps=[
                QueryStep(step_id=1, description="Success", database="postgresql",
                         query="SELECT id FROM users", depends_on=[]),
                QueryStep(step_id=2, description="Fail", database="postgresql",
                         query="SELECT FAIL FROM nowhere", depends_on=[]),
            ],
            reasoning="One success, one failure",
        )

        result = await executor.execute(plan)

        assert result.success is False
        assert 1 in result.step_results  # First query succeeded
        assert 2 in result.errors  # Second query failed
        assert "Database connection lost" in result.errors[2]


class TestIdentityResolutionIntegration:
    """Test identity resolution across the system."""

    @pytest.mark.asyncio
    async def test_cross_database_query_with_identity(self, schema, temp_identity_path):
        """Test querying across databases using identity resolution."""
        # Setup mapping store with links
        mapping_store = MappingStore(temp_identity_path)
        
        # User 1 exists in both PG and Qdrant
        mapping_store.save_link(IdentityLink(
            canonical_id="user:alice",
            source="postgresql",
            source_key="1",
            match_rule="primary",
            confidence=1.0,
        ))
        mapping_store.save_link(IdentityLink(
            canonical_id="user:alice",
            source="qdrant",
            source_key="vec_alice",
            match_rule="email_match",
            confidence=0.95,
        ))

        # Mock adapters
        mock_pg = MagicMock()
        mock_pg.database_type = DatabaseType.POSTGRESQL
        mock_pg.execute = AsyncMock(return_value=QueryResult(
            columns=["id", "name", "email"],
            rows=[[1, "Alice", "alice@example.com"]],
            row_count=1,
        ))

        mock_qdrant = MagicMock()
        mock_qdrant.database_type = DatabaseType.QDRANT

        executor = QueryExecutor(
            adapters={"postgresql": mock_pg, "qdrant": mock_qdrant},
            schema=schema,
            mapping_store=mapping_store,
        )

        # Query that needs identity resolution
        plan = QueryPlan(
            steps=[
                QueryStep(
                    step_id=1,
                    description="Get user from PostgreSQL",
                    database="postgresql",
                    query="SELECT id, name, email FROM users WHERE id = 1",
                    depends_on=[],
                ),
                QueryStep(
                    step_id=2,
                    step_type=StepType.RESOLVE_IDENTITY,
                    description="Resolve to canonical ID",
                    database="postgresql",
                    mapping_name="user",  # Entity name
                    input_from=1,
                    input_key="id",
                    depends_on=[1],
                ),
            ],
            explanation="Get user and resolve identity for cross-database lookup",
        )

        result = await executor.execute(plan)

        assert result.success is True
        
        # Check identity was resolved
        identity_result = result.identity_results[2]
        assert "user:alice" in identity_result.canonical_ids
        
        # Check we can get the qdrant key for alice
        qdrant_keys = mapping_store.get_source_keys(["user:alice"], "qdrant")
        assert qdrant_keys["user:alice"] == "vec_alice"


class TestSchemaLoaderIntegration:
    """Test schema loading and validation."""

    def test_load_schema_with_identity(self, schema):
        """Test loading schema with identity configuration."""
        user_entity = schema.get_entity("user")
        assert user_entity is not None
        assert user_entity.identity is not None
        assert user_entity.identity.canonical_id == "user_id"
        assert len(user_entity.identity.sources) == 2

    def test_get_cross_database_mappings(self, schema):
        """Test getting cross-database mappings."""
        assert len(schema.cross_database_mappings) == 1
        mapping = schema.cross_database_mappings[0]
        assert mapping.name == "user_unified_view"
        assert len(mapping.sources) == 2


class TestEndToEndQueryFlow:
    """End-to-end tests simulating real query scenarios."""

    @pytest.mark.asyncio
    async def test_execute_and_interpret_flow(self, schema, mapping_store):
        """Test execution and interpretation flow with pre-built plan."""
        # Mock adapter
        mock_pg = MagicMock()
        mock_pg.database_type = DatabaseType.POSTGRESQL
        mock_pg.execute = AsyncMock(return_value=QueryResult(
            columns=["id", "name", "total"],
            rows=[
                [1, "Alice", 5000.0],
                [2, "Bob", 3500.0],
                [3, "Charlie", 2000.0],
            ],
            row_count=3,
        ))

        # Create executor
        executor = QueryExecutor(
            adapters={"postgresql": mock_pg},
            schema=schema,
            mapping_store=mapping_store,
        )

        # Build plan directly (simulating what planner would generate)
        plan = QueryPlan(
            steps=[
                QueryStep(
                    step_id=1,
                    description="Get top spending users",
                    database="postgresql",
                    query="SELECT u.id, u.name, SUM(o.amount) as total FROM users u JOIN orders o ON u.id = o.user_id GROUP BY u.id, u.name ORDER BY total DESC LIMIT 10",
                    depends_on=[],
                )
            ],
            explanation="Join users with orders to calculate spending",
        )

        # Execute
        exec_result = await executor.execute(plan)
        assert exec_result.success is True
        assert 1 in exec_result.step_results
        
        # Verify results
        result = exec_result.step_results[1]
        assert result.row_count == 3
        assert result.columns == ["id", "name", "total"]
        
        # Verify data
        data = result.to_dicts()
        assert data[0]["name"] == "Alice"
        assert data[0]["total"] == 5000.0
        assert data[1]["name"] == "Bob"
        assert data[2]["name"] == "Charlie"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
