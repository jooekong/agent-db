"""Models for identity resolution and mapping."""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from agent_db.semantic.models import MatchStrategy


class MatchRule(BaseModel):
    """Matching rule definition."""

    name: str
    strategy: MatchStrategy = MatchStrategy.EXACT
    fields: list[str]
    threshold: Optional[float] = None
    confidence: Optional[float] = None


class IdentityLink(BaseModel):
    """Link between canonical ID and a source record."""

    canonical_id: str
    source: str
    source_key: str
    match_rule: str
    confidence: float = Field(ge=0.0, le=1.0)
    provenance: dict[str, str] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class ResolutionStatus(str, Enum):
    """Resolution job status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ResolutionJob(BaseModel):
    """Identity resolution job tracking."""

    job_id: str
    entity: str
    status: ResolutionStatus = ResolutionStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
