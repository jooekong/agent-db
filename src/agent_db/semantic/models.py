"""Core models for static semantic layer."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class SemanticType(str, Enum):
    """Entity semantic types."""

    ACTOR = "actor"
    OBJECT = "object"
    EVENT = "event"
    METRIC = "metric"


class AttributeSemanticType(str, Enum):
    """Attribute semantic types."""

    DIMENSION = "dimension"
    MEASURE = "measure"
    IDENTIFIER = "identifier"
    TIMESTAMP = "timestamp"
    TEXT = "text"


class Lifecycle(BaseModel):
    """Entity lifecycle column mappings."""

    created: str
    updated: Optional[str] = None
    deleted: Optional[str] = None


class EntityState(BaseModel):
    """Named state with SQL condition."""

    name: str
    condition: str


class EnumValue(BaseModel):
    """Enumeration value with business meaning."""

    value: str
    meaning: str
    business_priority: Optional[str] = None


class Attribute(BaseModel):
    """Column-level semantic definition."""

    column: str
    semantic_type: AttributeSemanticType
    description: Optional[str] = None
    enum_values: list[EnumValue] = Field(default_factory=list)


class Entity(BaseModel):
    """Table-level semantic definition."""

    name: str
    table: str
    description: str
    semantic_type: SemanticType
    lifecycle: Optional[Lifecycle] = None
    states: list[EntityState] = Field(default_factory=list)
    attributes: list[Attribute] = Field(default_factory=list)


class DatabaseRole(str, Enum):
    """Role of a database in cross-database mapping."""

    MASTER = "master"
    ENRICHMENT = "enrichment"


class DataSource(BaseModel):
    """A data source in cross-database mapping."""

    database: str
    entity: Optional[str] = None
    collection: Optional[str] = None
    node: Optional[str] = None
    role: DatabaseRole
    provides: list[str] = Field(default_factory=list)


class QueryRouting(BaseModel):
    """Query routing rule based on pattern matching."""

    pattern: str
    prefer: str


class CrossDatabaseMapping(BaseModel):
    """Unified view across multiple databases."""

    name: str
    sources: list[DataSource]
    query_routing: list[QueryRouting] = Field(default_factory=list)
