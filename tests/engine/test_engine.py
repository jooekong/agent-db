"""Tests for query engine components."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_db.engine.intent import IntentParser, ParsedIntent, IntentType
from agent_db.engine.planner import QueryPlanner, QueryPlan, QueryStep
from agent_db.engine.executor import QueryExecutor, ExecutionResult
from agent_db.engine.interpreter import ResultInterpreter, InterpretedResult
from agent_db.semantic.loader import SemanticSchema
from agent_db.semantic.models import Entity, SemanticType
from agent_db.adapters.protocol import QueryResult
from agent_db.llm.provider import LLMConfig


@pytest.fixture
def llm_config() -> LLMConfig:
    return LLMConfig(model="gpt-4")


@pytest.fixture
def schema() -> SemanticSchema:
    return SemanticSchema(
        entities=[
            Entity(
                name="user",
                table="users",
                description="System users",
                semantic_type=SemanticType.ACTOR,
            ),
        ]
    )


class TestIntentParser:
    @pytest.mark.asyncio
    async def test_parse_basic_query(self, llm_config: LLMConfig):
        parser = IntentParser(llm_config)

        mock_complete = AsyncMock(return_value={
            "type": "aggregation",
            "subject": "users",
            "timeframe": None,
            "filters": {},
            "entities": ["user"],
        })
        parser.llm.complete_json = mock_complete

        intent = await parser.parse("How many users?")

        assert intent.type == IntentType.AGGREGATION
        assert "user" in intent.entities


class TestQueryPlanner:
    @pytest.mark.asyncio
    async def test_plan_simple_query(self, llm_config: LLMConfig, schema: SemanticSchema):
        planner = QueryPlanner(llm_config, schema)

        intent = ParsedIntent(
            type=IntentType.AGGREGATION,
            subject="users",
            entities=["user"],
            raw_query="How many users?",
        )

        mock_complete = AsyncMock(return_value={
            "steps": [
                {
                    "step_id": 1,
                    "database": "postgresql",
                    "query": "SELECT COUNT(*) FROM users",
                    "description": "Count users",
                }
            ]
        })
        planner.llm.complete_json = mock_complete

        plan = await planner.plan(intent)

        assert len(plan.steps) == 1
        assert plan.steps[0].database == "postgresql"


class TestQueryExecutor:
    @pytest.fixture
    def mock_adapter(self):
        adapter = AsyncMock()
        adapter.execute.return_value = QueryResult(
            columns=["count"], rows=[[100]], row_count=1
        )
        return adapter

    @pytest.mark.asyncio
    async def test_execute_single_step(self, mock_adapter):
        executor = QueryExecutor({"postgresql": mock_adapter})

        plan = QueryPlan(
            steps=[
                QueryStep(
                    step_id=1,
                    database="postgresql",
                    query="SELECT COUNT(*) FROM users",
                    description="Count users",
                )
            ]
        )

        result = await executor.execute(plan)

        assert len(result.step_results) == 1
        assert result.step_results[1].row_count == 1

    @pytest.mark.asyncio
    async def test_execute_with_dependencies(self, mock_adapter):
        executor = QueryExecutor({"postgresql": mock_adapter})

        plan = QueryPlan(
            steps=[
                QueryStep(
                    step_id=1,
                    database="postgresql",
                    query="SELECT id FROM users",
                    description="Get IDs",
                ),
                QueryStep(
                    step_id=2,
                    database="postgresql",
                    query="SELECT * FROM orders",
                    description="Get orders",
                    depends_on=[1],
                ),
            ]
        )

        result = await executor.execute(plan)

        assert len(result.step_results) == 2
        assert result.success


class TestResultInterpreter:
    @pytest.mark.asyncio
    async def test_interpret_result(self, llm_config: LLMConfig):
        interpreter = ResultInterpreter(llm_config)

        intent = ParsedIntent(
            type=IntentType.AGGREGATION,
            subject="users",
            entities=["user"],
            raw_query="How many users?",
        )

        execution_result = ExecutionResult(
            step_results={
                1: QueryResult(columns=["count"], rows=[[1500]], row_count=1)
            }
        )

        mock_complete = AsyncMock(return_value="You have 1,500 users in the system.")
        interpreter.llm.complete = mock_complete

        result = await interpreter.interpret(intent, execution_result)

        assert "1,500" in result.summary or "1500" in result.summary
