"""YAML schema loader for semantic layer."""

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field

from agent_db.semantic.models import (
    Entity,
    CrossDatabaseMapping,
)


class SemanticSchema(BaseModel):
    """Complete semantic schema definition."""

    entities: list[Entity] = Field(default_factory=list)
    cross_database_mappings: list[CrossDatabaseMapping] = Field(default_factory=list)

    def get_entity(self, name: str) -> Optional[Entity]:
        """Get entity by name."""
        for entity in self.entities:
            if entity.name == name:
                return entity
        return None

    def get_mapping(self, name: str) -> Optional[CrossDatabaseMapping]:
        """Get cross-database mapping by name."""
        for mapping in self.cross_database_mappings:
            if mapping.name == name:
                return mapping
        return None


class SchemaLoader:
    """Loads semantic schema from YAML files."""

    def load(self, path: Path) -> SemanticSchema:
        """Load schema from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return self._parse(data)

    def load_from_string(self, content: str) -> SemanticSchema:
        """Load schema from YAML string."""
        data = yaml.safe_load(content)
        return self._parse(data)

    def _parse(self, data: dict) -> SemanticSchema:
        """Parse raw dict into SemanticSchema."""
        if data is None:
            return SemanticSchema()
        return SemanticSchema(**data)
