"""Persistent storage for dynamic metadata."""

import json
from pathlib import Path
from typing import Optional

from agent_db.metadata.models import (
    DataProfile,
    CorrelationInsight,
    QueryPattern,
)


class MetadataStore:
    """File-based metadata storage."""

    def __init__(self, base_path: Path):
        self.base_path = base_path
        self._profiles_path = base_path / "profiles"
        self._insights_path = base_path / "insights"
        self._patterns_path = base_path / "patterns"

        # Ensure directories exist
        for path in [self._profiles_path, self._insights_path, self._patterns_path]:
            path.mkdir(parents=True, exist_ok=True)

    def save_profile(self, profile: DataProfile) -> None:
        """Save data profile."""
        path = self._profiles_path / f"{profile.table}.json"
        path.write_text(profile.model_dump_json(indent=2))

    def get_profile(self, table: str) -> Optional[DataProfile]:
        """Get data profile by table name."""
        path = self._profiles_path / f"{table}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        return DataProfile(**data)

    def list_profiles(self) -> list[str]:
        """List all profile table names."""
        return [p.stem for p in self._profiles_path.glob("*.json")]

    def save_insight(self, insight: CorrelationInsight) -> None:
        """Save correlation insight."""
        path = self._insights_path / f"{insight.insight_id}.json"
        path.write_text(insight.model_dump_json(indent=2))

    def get_insight(self, insight_id: str) -> Optional[CorrelationInsight]:
        """Get insight by ID."""
        path = self._insights_path / f"{insight_id}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        return CorrelationInsight(**data)

    def list_insights(self) -> list[str]:
        """List all insight IDs."""
        return [p.stem for p in self._insights_path.glob("*.json")]

    def save_query_pattern(self, pattern: QueryPattern) -> None:
        """Save query pattern."""
        path = self._patterns_path / f"{pattern.pattern_id}.json"
        path.write_text(pattern.model_dump_json(indent=2))

    def get_query_pattern(self, pattern_id: str) -> Optional[QueryPattern]:
        """Get query pattern by ID."""
        path = self._patterns_path / f"{pattern_id}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        return QueryPattern(**data)

    def list_query_patterns(self) -> list[str]:
        """List all pattern IDs."""
        return [p.stem for p in self._patterns_path.glob("*.json")]
