"""Query executor for running query plans."""

from pydantic import BaseModel, Field

from agent_db.engine.planner import QueryPlan, QueryStep
from agent_db.adapters.protocol import DatabaseAdapter, QueryResult


class ExecutionResult(BaseModel):
    """Result of query plan execution."""

    step_results: dict[int, QueryResult] = Field(default_factory=dict)
    errors: dict[int, str] = Field(default_factory=dict)
    success: bool = True


class QueryExecutor:
    """Executes query plans across databases."""

    def __init__(self, adapters: dict[str, DatabaseAdapter]):
        self.adapters = adapters

    async def execute(self, plan: QueryPlan) -> ExecutionResult:
        """Execute query plan."""
        result = ExecutionResult()
        completed: set[int] = set()

        # Sort steps by dependencies
        pending = list(plan.steps)

        while pending:
            # Find steps with satisfied dependencies
            ready = [
                s for s in pending
                if all(d in completed for d in s.depends_on)
            ]

            if not ready:
                # Circular dependency or missing dependency
                for s in pending:
                    result.errors[s.step_id] = "Unresolved dependencies"
                result.success = False
                break

            for step in ready:
                try:
                    step_result = await self._execute_step(step, result.step_results)
                    result.step_results[step.step_id] = step_result
                    completed.add(step.step_id)
                except Exception as e:
                    result.errors[step.step_id] = str(e)
                    result.success = False

                pending.remove(step)

        return result

    async def _execute_step(
        self,
        step: QueryStep,
        previous_results: dict[int, QueryResult],
    ) -> QueryResult:
        """Execute a single query step."""
        adapter = self.adapters.get(step.database)
        if not adapter:
            raise ValueError(f"No adapter for database: {step.database}")

        return await adapter.execute(step.query)
