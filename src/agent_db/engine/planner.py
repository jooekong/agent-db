"""Query planner for generating execution plans."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from agent_db.engine.intent import ParsedIntent
from agent_db.semantic.loader import SemanticSchema
from agent_db.llm.provider import LLMProvider, LLMConfig, Message, Role


class StepType(str, Enum):
    """Types of query plan steps."""

    QUERY = "query"
    RESOLVE_IDENTITY = "resolve_identity"


class QueryStep(BaseModel):
    """A single step in query execution plan."""

    step_id: int
    database: str
    query: str = ""  # Empty for RESOLVE_IDENTITY steps
    description: str
    depends_on: list[int] = Field(default_factory=list)
    step_type: StepType = StepType.QUERY
    mapping_name: Optional[str] = None  # For RESOLVE_IDENTITY: cross-db mapping name
    input_from: Optional[int] = None  # For RESOLVE_IDENTITY: step to get IDs from
    input_key: Optional[str] = None  # For RESOLVE_IDENTITY: key column name


class QueryPlan(BaseModel):
    """Complete query execution plan."""

    steps: list[QueryStep]
    explanation: str = ""


PLANNER_SYSTEM_PROMPT = """You are a query planner. Given:
1. A parsed intent
2. Available schema
3. Data profiles (statistics about each table)
4. Identity mappings for cross-database linking

Generate an execution plan with SQL/Cypher/Flux queries.

Schema:
{schema}

Data Profiles (use this to optimize queries):
{profiles}

Identity mappings (use resolve_identity steps to join across sources):
{identity_mappings}

Optimization hints:
- For high-cardinality columns, use LIMIT
- For long-tail distributions, consider percentile-based filtering
- For uniform distributions, range queries work well
- Row counts help decide between full scan vs sampling

Respond in JSON with:
{{
  "steps": [
    {{
      "step_id": 1,
      "database": "postgresql|qdrant|neo4j|influxdb",
      "query": "actual query",
      "description": "what this step does",
      "depends_on": [],
      "step_type": "query|resolve_identity",
      "mapping_name": "entity_name_for_identity",
      "input_from": 1,
      "input_key": "column_name_from_step"
    }}
  ]
}}"""


class QueryPlanner:
    """Plans query execution across databases."""

    def __init__(self, config: LLMConfig, schema: SemanticSchema):
        self.llm = LLMProvider(config)
        self.schema = schema
        self._profile_context: str = ""

    def set_profile_context(self, profile_context: str) -> None:
        """Set data profile context for query optimization."""
        self._profile_context = profile_context

    async def plan(self, intent: ParsedIntent) -> QueryPlan:
        """Generate execution plan for intent."""
        schema_desc = self._format_schema()
        profiles_desc = self._profile_context or "No profiles available"
        identity_desc = self._format_identity_mappings() or "No identity mappings available"

        messages = [
            Message(
                role=Role.SYSTEM,
                content=PLANNER_SYSTEM_PROMPT.format(
                    schema=schema_desc,
                    profiles=profiles_desc,
                    identity_mappings=identity_desc,
                ),
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

    def _format_identity_mappings(self) -> str:
        """Format identity mappings for LLM context."""
        lines = []
        for entity in self.schema.entities:
            identity = entity.identity
            if not identity:
                continue
            sources = ", ".join(
                f"{s.database}:{s.key_column}" for s in identity.sources
            )
            lines.append(f"- {entity.name}: canonical_id={identity.canonical_id}, sources=[{sources}]")
            if identity.match_rules:
                rules = ", ".join(f"{r.name}:{r.strategy.value}" for r in identity.match_rules)
                lines.append(f"  match_rules=[{rules}]")
        return "\n".join(lines)
