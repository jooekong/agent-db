"""Tests for dynamic metadata models."""

from datetime import datetime

import pytest

from agent_db.metadata.models import (
    DataProfile,
    ColumnStats,
    Distribution,
    DistributionType,
    CorrelationInsight,
    QueryPattern,
)


class TestDataProfile:
    def test_profile_basic(self):
        profile = DataProfile(
            table="orders",
            last_updated=datetime(2024, 1, 15, 10, 0, 0),
            row_count=12_500_000,
        )
        assert profile.table == "orders"
        assert profile.row_count == 12_500_000

    def test_profile_with_column_stats(self):
        stats = ColumnStats(
            name="amount",
            min_val=0.01,
            max_val=99999.99,
            avg_val=156.78,
            median_val=89.00,
            distribution=Distribution(
                type=DistributionType.LONG_TAIL,
                p25=45.00,
                p75=180.00,
                p99=1200.00,
            ),
        )
        profile = DataProfile(
            table="orders",
            last_updated=datetime.now(),
            row_count=1000,
            columns=[stats],
        )
        assert len(profile.columns) == 1
        assert profile.columns[0].avg_val == 156.78


class TestCorrelationInsight:
    def test_insight_basic(self):
        insight = CorrelationInsight(
            insight_id="corr_001",
            confidence=0.92,
            description="High value users tend to order on weekends",
            evidence="High value users weekend order ratio 45%, normal users only 28%",
            usage_hint="Mention when asked about high value user behavior",
        )
        assert insight.confidence == 0.92


class TestQueryPattern:
    def test_pattern_basic(self):
        pattern = QueryPattern(
            pattern_id="qp_001",
            question_type="User segmentation analysis",
            example_questions=[
                "Which users might churn",
                "What characterizes high value users",
            ],
            query_template="WITH user_segments AS (...) SELECT ...",
            usage_count=156,
        )
        assert pattern.usage_count == 156
        assert len(pattern.example_questions) == 2
