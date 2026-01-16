"""Result interpreter for generating natural language responses."""

from pydantic import BaseModel, Field

from agent_db.engine.intent import ParsedIntent
from agent_db.engine.executor import ExecutionResult
from agent_db.llm.provider import LLMProvider, LLMConfig, Message, Role


class InterpretedResult(BaseModel):
    """Interpreted query result."""

    summary: str
    suggestions: list[str] = Field(default_factory=list)
    raw_data: dict = Field(default_factory=dict)


INTERPRETER_SYSTEM_PROMPT = """You are a data analyst assistant. Given:
1. The user's original question
2. Query results

Provide a clear, natural language response that:
1. Directly answers the question
2. Highlights key insights
3. Suggests follow-up questions if relevant

Be concise but informative."""


class ResultInterpreter:
    """Interprets query results into natural language."""

    def __init__(self, config: LLMConfig):
        self.llm = LLMProvider(config)

    async def interpret(
        self,
        intent: ParsedIntent,
        result: ExecutionResult,
    ) -> InterpretedResult:
        """Interpret execution results."""
        results_text = self._format_results(result)

        messages = [
            Message(role=Role.SYSTEM, content=INTERPRETER_SYSTEM_PROMPT),
            Message(
                role=Role.USER,
                content=f"Question: {intent.raw_query}\n\nResults:\n{results_text}",
            ),
        ]

        summary = await self.llm.complete(messages)
        suggestions = self._extract_suggestions(summary)

        return InterpretedResult(
            summary=summary,
            suggestions=suggestions,
            raw_data={"step_results": {k: v.model_dump() for k, v in result.step_results.items()}},
        )

    def _format_results(self, result: ExecutionResult) -> str:
        """Format execution results for LLM."""
        lines = []
        for step_id, query_result in result.step_results.items():
            lines.append(f"Step {step_id}:")
            lines.append(f"  Columns: {query_result.columns}")
            lines.append(f"  Rows: {query_result.rows[:10]}")
            if query_result.row_count > 10:
                lines.append(f"  ... and {query_result.row_count - 10} more rows")
        return "\n".join(lines)

    def _extract_suggestions(self, text: str) -> list[str]:
        """Extract suggestions from LLM response."""
        suggestions = []
        lines = text.split("\n")
        in_suggestions = False

        for line in lines:
            if "suggestion" in line.lower() or "follow-up" in line.lower():
                in_suggestions = True
                continue
            if in_suggestions and line.strip().startswith("-"):
                suggestions.append(line.strip()[1:].strip())

        return suggestions
