"""Tests for metadata store."""

from datetime import datetime
from pathlib import Path
import tempfile

import pytest

from agent_db.metadata.store import MetadataStore
from agent_db.metadata.models import (
    DataProfile,
    ColumnStats,
    CorrelationInsight,
    QueryPattern,
)


@pytest.fixture
def temp_store_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "metadata"


@pytest.fixture
def sample_profile() -> DataProfile:
    return DataProfile(
        table="orders",
        last_updated=datetime(2024, 1, 15),
        row_count=1000,
        columns=[
            ColumnStats(name="amount", avg_val=100.0),
        ],
    )


class TestMetadataStore:
    def test_save_and_load_profile(self, temp_store_path: Path, sample_profile: DataProfile):
        store = MetadataStore(temp_store_path)
        store.save_profile(sample_profile)

        loaded = store.get_profile("orders")
        assert loaded is not None
        assert loaded.table == "orders"
        assert loaded.row_count == 1000

    def test_get_nonexistent_profile(self, temp_store_path: Path):
        store = MetadataStore(temp_store_path)
        result = store.get_profile("nonexistent")
        assert result is None

    def test_save_and_load_insight(self, temp_store_path: Path):
        store = MetadataStore(temp_store_path)
        insight = CorrelationInsight(
            insight_id="test_001",
            confidence=0.85,
            description="Test insight",
            evidence="Test evidence",
            usage_hint="Test hint",
        )
        store.save_insight(insight)

        loaded = store.get_insight("test_001")
        assert loaded is not None
        assert loaded.confidence == 0.85

    def test_list_all_profiles(self, temp_store_path: Path, sample_profile: DataProfile):
        store = MetadataStore(temp_store_path)
        store.save_profile(sample_profile)

        profile2 = DataProfile(
            table="users",
            last_updated=datetime.now(),
            row_count=500,
        )
        store.save_profile(profile2)

        profiles = store.list_profiles()
        assert len(profiles) == 2
        assert "orders" in profiles
        assert "users" in profiles

    def test_save_and_load_query_pattern(self, temp_store_path: Path):
        store = MetadataStore(temp_store_path)
        pattern = QueryPattern(
            pattern_id="qp_001",
            question_type="Aggregation",
            example_questions=["How many users"],
            query_template="SELECT COUNT(*) FROM users",
            usage_count=10,
        )
        store.save_query_pattern(pattern)

        loaded = store.get_query_pattern("qp_001")
        assert loaded is not None
        assert loaded.usage_count == 10
