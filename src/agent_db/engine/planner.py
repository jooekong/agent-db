"""Query planner for generating execution plans."""

from pydantic import BaseModel, Field

from agent_db.engine.intent import ParsedIntent
from agent_db.semantic.loader import SemanticSchema
from agent_db.llm.provider import LLMProvider, LLMConfig, Message, Role


class QueryStep(BaseModel):
    """A single step in query execution plan."""

    step_id: int
    database: str
    query: str
    description: str
    depends_on: list[int] = Field(default_factory=list)


class QueryPlan(BaseModel):
    """Complete query execution plan."""

    steps: list[QueryStep]
    explanation: str = ""


PLANNER_SYSTEM_PROMPT = """You are a query planner. Given:
1. A parsed intent
2. Available schema

Generate an execution plan with SQL/Cypher/Flux queries.

Schema:
{schema}

Respond in JSON with:
{{
  "steps": [
    {{
      "step_id": 1,
      "database": "postgresql|qdrant|neo4j|influxdb",
      "query": "actual query",
      "description": "what this step does",
      "depends_on": []
    }}
  ]
}}"""


class QueryPlanner:
    """Plans query execution across databases."""

    def __init__(self, config: LLMConfig, schema: SemanticSchema):
        self.llm = LLMProvider(config)
        self.schema = schema

    async def plan(self, intent: ParsedIntent) -> QueryPlan:
        """Generate execution plan for intent."""
        schema_desc = self._format_schema()

        messages = [
            Message(
                role=Role.SYSTEM,
                content=PLANNER_SYSTEM_PROMPT.format(schema=schema_desc),
            ),
            Message(role=Role.USER, content=intent.model_dump_json()),
        ]

        result = await self.llm.complete_json(messages)

        steps = [QueryStep(**s) for s in result["steps"]]
        return QueryPlan(steps=steps)

    def _format_schema(self) -> str:
        """Format schema for LLM context."""
        lines = []
        for entity in self.schema.entities:
            lines.append(f"- {entity.name}: table={entity.table}, type={entity.semantic_type}")
        return "\n".join(lines)
