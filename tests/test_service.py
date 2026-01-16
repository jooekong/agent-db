"""Tests for query service."""

from unittest.mock import AsyncMock, MagicMock
from pathlib import Path
import tempfile

import pytest

from agent_db.service import QueryService, ServiceConfig
from agent_db.engine.interpreter import InterpretedResult
from agent_db.adapters.protocol import QueryResult


@pytest.fixture
def temp_config():
    with tempfile.TemporaryDirectory() as tmpdir:
        schema_path = Path(tmpdir) / "schema.yaml"
        schema_path.write_text("""
entities:
  - name: user
    table: users
    description: "System users"
    semantic_type: actor
""")
        yield ServiceConfig(
            schema_path=schema_path,
            metadata_path=Path(tmpdir) / "metadata",
            llm_model="gpt-4",
        )


class TestQueryService:
    def test_service_initialization(self, temp_config: ServiceConfig):
        service = QueryService(temp_config)
        assert service.schema is not None
        assert len(service.schema.entities) == 1

    @pytest.mark.asyncio
    async def test_query_flow(self, temp_config: ServiceConfig):
        service = QueryService(temp_config)

        # Mock LLM calls
        service.intent_parser.llm.complete_json = AsyncMock(return_value={
            "type": "aggregation",
            "subject": "users",
            "entities": ["user"],
        })
        service.planner.llm.complete_json = AsyncMock(return_value={
            "steps": [{
                "step_id": 1,
                "database": "postgresql",
                "query": "SELECT COUNT(*) FROM users",
                "description": "Count users",
            }]
        })
        service.interpreter.llm.complete = AsyncMock(
            return_value="You have 1000 users."
        )

        # Add mock adapter
        mock_adapter = AsyncMock()
        mock_adapter.execute.return_value = QueryResult(
            columns=["count"],
            rows=[[1000]],
            row_count=1,
        )
        service.add_adapter("postgresql", mock_adapter)

        result = await service.query("How many users?")

        assert isinstance(result, InterpretedResult)
        assert "1000" in result.summary
