"""Query executor for running query plans."""

import asyncio
from typing import Any, Optional

from pydantic import BaseModel, Field

from agent_db.engine.planner import QueryPlan, QueryStep, StepType
from agent_db.adapters.protocol import DatabaseAdapter, QueryResult
from agent_db.identity.store import MappingStore
from agent_db.semantic.loader import SemanticSchema


class IdentityResolutionResult(BaseModel):
    """Resolved identity mappings for a step."""

    canonical_ids: list[str] = Field(default_factory=list)
    source_keys: dict[str, list[str]] = Field(default_factory=dict)


class ExecutionResult(BaseModel):
    """Result of query plan execution."""

    step_results: dict[int, QueryResult] = Field(default_factory=dict)
    identity_results: dict[int, IdentityResolutionResult] = Field(default_factory=dict)
    errors: dict[int, str] = Field(default_factory=dict)
    success: bool = True


class QueryExecutor:
    """Executes query plans across databases."""

    def __init__(
        self,
        adapters: dict[str, DatabaseAdapter],
        schema: Optional[SemanticSchema] = None,
        mapping_store: Optional[MappingStore] = None,
    ):
        self.adapters = adapters
        self.schema = schema
        self.mapping_store = mapping_store

    async def execute(self, plan: QueryPlan) -> ExecutionResult:
        """Execute query plan."""
        result = ExecutionResult()
        completed: set[int] = set()
        step_by_id = {step.step_id: step for step in plan.steps}

        pending = list(plan.steps)

        while pending:
            ready = [
                s for s in pending
                if all(d in completed for d in s.depends_on)
            ]

            if not ready:
                for s in pending:
                    result.errors[s.step_id] = "Unresolved dependencies"
                result.success = False
                break

            # Separate identity resolution (sync) from query steps (async)
            identity_steps = [s for s in ready if s.step_type == StepType.RESOLVE_IDENTITY]
            query_steps = [s for s in ready if s.step_type != StepType.RESOLVE_IDENTITY]

            # Execute identity resolution steps first (synchronous)
            for step in identity_steps:
                try:
                    identity_result = self._resolve_identity_step(
                        step, result.step_results, step_by_id
                    )
                    result.identity_results[step.step_id] = identity_result
                except Exception as e:
                    result.errors[step.step_id] = str(e)
                    result.success = False
                # Always mark as completed to maintain dependency tracking
                completed.add(step.step_id)
                pending.remove(step)

            # Execute query steps in parallel
            if query_steps:
                tasks = [
                    self._execute_step_safe(
                        step, result.step_results, result.identity_results, step_by_id
                    )
                    for step in query_steps
                ]
                step_outcomes = await asyncio.gather(*tasks)

                for step, outcome in zip(query_steps, step_outcomes):
                    if isinstance(outcome, Exception):
                        result.errors[step.step_id] = str(outcome)
                        result.success = False
                    else:
                        result.step_results[step.step_id] = outcome
                    completed.add(step.step_id)
                    pending.remove(step)

        return result

    async def _execute_step_safe(
        self,
        step: QueryStep,
        previous_results: dict[int, QueryResult],
        identity_results: dict[int, "IdentityResolutionResult"],
        step_by_id: dict[int, QueryStep],
    ) -> QueryResult | Exception:
        """Execute step, returning exception on failure instead of raising."""
        try:
            return await self._execute_step(step, previous_results, identity_results, step_by_id)
        except Exception as e:
            return e

    async def _execute_step(
        self,
        step: QueryStep,
        previous_results: dict[int, QueryResult],
        identity_results: dict[int, IdentityResolutionResult],
        step_by_id: dict[int, QueryStep],
    ) -> QueryResult:
        """Execute a single query step."""
        adapter = self.adapters.get(step.database)
        if not adapter:
            raise ValueError(f"No adapter for database: {step.database}")

        query = step.query
        context = self._build_query_context(step, identity_results, step_by_id)
        if context:
            query = self._render_query(query, context)

        return await adapter.execute(query)

    def _resolve_identity_step(
        self,
        step: QueryStep,
        previous_results: dict[int, QueryResult],
        step_by_id: dict[int, QueryStep],
    ) -> IdentityResolutionResult:
        """Resolve identities using the mapping store."""
        if not self.mapping_store or not self.schema:
            raise ValueError("Identity mapping store or schema not configured")
        if step.mapping_name is None:
            raise ValueError("Identity resolution step missing mapping_name")
        if step.input_from is None or step.input_key is None:
            raise ValueError("Identity resolution step missing input_from/input_key")

        input_step = step_by_id.get(step.input_from)
        if not input_step:
            raise ValueError(f"Unknown input step: {step.input_from}")
        input_result = previous_results.get(step.input_from)
        if not input_result:
            raise ValueError(f"Missing results for input step: {step.input_from}")

        if step.input_key not in input_result.columns:
            raise ValueError(f"Column not found in input results: {step.input_key}")

        idx = input_result.columns.index(step.input_key)
        source_keys = [row[idx] for row in input_result.rows if row[idx] is not None]
        source_keys_str = [str(key) for key in source_keys]

        canonical_map = self.mapping_store.get_canonical_ids(
            input_step.database, source_keys_str
        )
        for key in source_keys_str:
            if key not in canonical_map:
                canonical_map[key] = f"{input_step.database}:{key}"

        canonical_ids = sorted(set(canonical_map.values()))

        entity = self.schema.get_entity(step.mapping_name)
        if not entity or not entity.identity:
            raise ValueError(f"Identity mapping not found for entity: {step.mapping_name}")

        source_keys_by_db: dict[str, list[str]] = {}
        for source in entity.identity.sources:
            mapping = self.mapping_store.get_source_keys(canonical_ids, source.database)
            source_keys_by_db[source.database] = list(mapping.values())

        return IdentityResolutionResult(
            canonical_ids=canonical_ids,
            source_keys=source_keys_by_db,
        )

    def _build_query_context(
        self,
        step: QueryStep,
        identity_results: dict[int, IdentityResolutionResult],
        step_by_id: dict[int, QueryStep],
    ) -> dict[str, list[str]]:
        context: dict[str, list[str]] = {}
        for dep_id in step.depends_on:
            if dep_id not in identity_results:
                continue
            identity_result = identity_results[dep_id]
            if identity_result.canonical_ids:
                context["canonical_ids"] = identity_result.canonical_ids
            if step.database in identity_result.source_keys:
                context["ids"] = identity_result.source_keys[step.database]
            for source, ids in identity_result.source_keys.items():
                context[f"ids_{source}"] = ids
        return context

    @staticmethod
    def _render_query(query: str, context: dict[str, list[str]]) -> str:
        rendered = query
        for key, values in context.items():
            rendered = rendered.replace(f"{{{key}}}", QueryExecutor._format_ids(values))
        return rendered

    @staticmethod
    def _format_ids(values: list[Any]) -> str:
        if not values:
            return "()"
        formatted = []
        for value in values:
            if isinstance(value, (int, float)):
                formatted.append(str(value))
            else:
                escaped = str(value).replace("'", "''")
                formatted.append(f"'{escaped}'")
        return f"({', '.join(formatted)})"
