"""Models for dynamic metadata layer."""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class DistributionType(str, Enum):
    """Statistical distribution types."""

    NORMAL = "normal"
    LONG_TAIL = "long_tail"
    UNIFORM = "uniform"
    BIMODAL = "bimodal"
    SKEWED = "skewed"


class Distribution(BaseModel):
    """Value distribution statistics."""

    type: DistributionType
    p25: Optional[float] = None
    p75: Optional[float] = None
    p99: Optional[float] = None


class ColumnStats(BaseModel):
    """Column-level statistics."""

    name: str
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    avg_val: Optional[float] = None
    median_val: Optional[float] = None
    null_ratio: Optional[float] = None
    distinct_count: Optional[int] = None
    distribution: Optional[Distribution] = None


class DataProfile(BaseModel):
    """Table-level data profile."""

    table: str
    last_updated: datetime
    row_count: int
    columns: list[ColumnStats] = Field(default_factory=list)


class CorrelationInsight(BaseModel):
    """AI-discovered correlation insight."""

    insight_id: str
    confidence: float = Field(ge=0.0, le=1.0)
    description: str
    evidence: str
    usage_hint: str


class QueryPattern(BaseModel):
    """Learned query pattern from user interactions."""

    pattern_id: str
    question_type: str
    example_questions: list[str]
    query_template: str
    usage_count: int = 0
