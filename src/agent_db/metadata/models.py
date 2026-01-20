"""Models for dynamic metadata layer."""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class DistributionType(str, Enum):
    """Statistical distribution types."""

    NORMAL = "normal"
    LONG_TAIL = "long_tail"
    UNIFORM = "uniform"
    BIMODAL = "bimodal"
    SKEWED = "skewed"
    UNKNOWN = "unknown"


class ColumnDataType(str, Enum):
    """Column data type categories."""

    NUMERIC = "numeric"
    TEXT = "text"
    TIMESTAMP = "timestamp"
    BOOLEAN = "boolean"
    UNKNOWN = "unknown"


class Distribution(BaseModel):
    """Value distribution statistics."""

    type: DistributionType
    p25: Optional[float] = None
    p75: Optional[float] = None
    p99: Optional[float] = None


class TextStats(BaseModel):
    """Text column specific statistics."""

    min_length: int = 0
    max_length: int = 0
    avg_length: float = 0.0
    empty_ratio: float = 0.0
    top_values: list[tuple[str, int]] = Field(default_factory=list)


class ColumnStats(BaseModel):
    """Column-level statistics."""

    name: str
    data_type: ColumnDataType = ColumnDataType.UNKNOWN
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    avg_val: Optional[float] = None
    median_val: Optional[float] = None
    null_ratio: Optional[float] = None
    distinct_count: Optional[int] = None
    distribution: Optional[Distribution] = None
    percentiles: Optional[dict[str, float]] = None
    text_stats: Optional[TextStats] = None


class DataProfile(BaseModel):
    """Table-level data profile."""

    table: str
    database: str = ""
    last_updated: datetime
    row_count: int
    columns: list[ColumnStats] = Field(default_factory=list)
    profile_version: int = 1
    schema_hash: str = ""


class ProfilingJobStatus(str, Enum):
    """Profiling job status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ProfilingJob(BaseModel):
    """Profiling job tracking."""

    job_id: str
    database: str
    table: str
    status: ProfilingJobStatus = ProfilingJobStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None


class ProfilingConfig(BaseModel):
    """Configuration for data profiling."""

    sample_size: int = 10000
    sample_method: str = "random"
    detect_distribution: bool = True
    calculate_percentiles: bool = True
    analyze_text: bool = True
    max_top_values: int = 10
    schedule_cron: Optional[str] = None
    extra: dict[str, Any] = Field(default_factory=dict)


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
