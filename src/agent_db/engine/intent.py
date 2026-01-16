"""Intent parser for natural language queries."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from agent_db.llm.provider import LLMProvider, LLMConfig, Message, Role


class IntentType(str, Enum):
    """Types of query intents."""

    AGGREGATION = "aggregation"
    TREND_ANALYSIS = "trend_analysis"
    COMPARISON = "comparison"
    LOOKUP = "lookup"
    RELATIONSHIP = "relationship"
    SIMILARITY = "similarity"


class ParsedIntent(BaseModel):
    """Parsed query intent."""

    type: IntentType
    subject: str
    timeframe: Optional[str] = None
    filters: dict = Field(default_factory=dict)
    entities: list[str] = Field(default_factory=list)
    raw_query: str = ""


INTENT_SYSTEM_PROMPT = """You are a query intent parser. Analyze the user's question and extract:

1. type: One of [aggregation, trend_analysis, comparison, lookup, relationship, similarity]
2. subject: The main subject being queried
3. timeframe: Time period mentioned (if any)
4. filters: Any conditions or qualifiers
5. entities: Database entities likely involved

Respond in JSON format only."""


class IntentParser:
    """Parses natural language queries into structured intents."""

    def __init__(self, config: LLMConfig):
        self.llm = LLMProvider(config)

    async def parse(self, query: str) -> ParsedIntent:
        """Parse a natural language query."""
        messages = [
            Message(role=Role.SYSTEM, content=INTENT_SYSTEM_PROMPT),
            Message(role=Role.USER, content=query),
        ]

        result = await self.llm.complete_json(messages)

        return ParsedIntent(
            type=IntentType(result["type"]),
            subject=result["subject"],
            timeframe=result.get("timeframe"),
            filters=result.get("filters", {}),
            entities=result.get("entities", []),
            raw_query=query,
        )
