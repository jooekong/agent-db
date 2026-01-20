"""Core profiling engine with distribution detection."""

import hashlib
from datetime import datetime
from typing import Any, Optional

import numpy as np
from scipy import stats

from agent_db.adapters.protocol import DatabaseAdapter, DatabaseType
from agent_db.metadata.models import (
    ColumnDataType,
    ColumnStats,
    DataProfile,
    Distribution,
    DistributionType,
    ProfilingConfig,
    TextStats,
)
from agent_db.profiling.sampler import BaseSampler, PostgreSQLSampler, get_sampler


class DistributionDetector:
    """Detects statistical distribution types from sample data."""

    SKEWNESS_THRESHOLD = 0.5
    HIGH_SKEWNESS_THRESHOLD = 1.0
    NORMAL_P_THRESHOLD = 0.05
    UNIFORM_P_THRESHOLD = 0.05

    @classmethod
    def detect(
        cls, values: list[float], percentiles: Optional[dict[str, float]] = None
    ) -> DistributionType:
        """Detect distribution type from sample values."""
        if not values or len(values) < 10:
            return DistributionType.UNKNOWN

        arr = np.array([v for v in values if v is not None])
        if len(arr) < 10:
            return DistributionType.UNKNOWN

        # Check for bimodal distribution first
        if cls._is_bimodal(arr):
            return DistributionType.BIMODAL

        # Calculate skewness and kurtosis
        skewness = stats.skew(arr)
        kurtosis = stats.kurtosis(arr)

        # Normality test (Shapiro-Wilk)
        sample_for_test = arr[:5000] if len(arr) > 5000 else arr
        try:
            _, p_normal = stats.shapiro(sample_for_test)
        except Exception:
            p_normal = 0

        # Uniformity test (Kolmogorov-Smirnov)
        normalized = (arr - arr.min()) / (arr.max() - arr.min() + 1e-10)
        try:
            _, p_uniform = stats.kstest(normalized, "uniform")
        except Exception:
            p_uniform = 0

        # Classification logic
        if p_uniform > cls.UNIFORM_P_THRESHOLD:
            return DistributionType.UNIFORM

        if p_normal > cls.NORMAL_P_THRESHOLD and abs(skewness) < cls.SKEWNESS_THRESHOLD:
            return DistributionType.NORMAL

        if skewness > cls.HIGH_SKEWNESS_THRESHOLD:
            return DistributionType.LONG_TAIL

        if abs(skewness) > cls.SKEWNESS_THRESHOLD:
            return DistributionType.SKEWED

        return DistributionType.NORMAL

    @classmethod
    def _is_bimodal(cls, arr: np.ndarray, bins: int = 50) -> bool:
        """Detect bimodal distribution using histogram peaks."""
        try:
            hist, bin_edges = np.histogram(arr, bins=bins)
            # Smooth histogram more aggressively
            kernel = np.array([1, 2, 3, 2, 1]) / 9
            smoothed = np.convolve(hist, kernel, mode="same")

            # Find peaks
            peaks = []
            for i in range(1, len(smoothed) - 1):
                if smoothed[i] > smoothed[i - 1] and smoothed[i] > smoothed[i + 1]:
                    peaks.append(i)

            # Filter significant peaks (> 40% of max, stricter threshold)
            if len(peaks) >= 2:
                max_height = max(smoothed[p] for p in peaks)
                significant_peaks = [p for p in peaks if smoothed[p] > max_height * 0.4]

                # Also check that peaks are well separated (at least 10 bins apart)
                if len(significant_peaks) >= 2:
                    significant_peaks.sort()
                    for i in range(len(significant_peaks) - 1):
                        if significant_peaks[i + 1] - significant_peaks[i] >= 10:
                            return True

        except Exception:
            pass

        return False


class ProfilerEngine:
    """Core engine for generating data profiles."""

    # PostgreSQL type mappings
    PG_NUMERIC_TYPES = {
        "integer", "bigint", "smallint", "decimal", "numeric",
        "real", "double precision", "serial", "bigserial", "money"
    }
    PG_TEXT_TYPES = {"character varying", "varchar", "character", "char", "text", "uuid"}
    PG_TIMESTAMP_TYPES = {"timestamp", "timestamptz", "date", "time", "timetz", "interval"}
    PG_BOOLEAN_TYPES = {"boolean"}

    def __init__(
        self,
        adapter: DatabaseAdapter,
        config: Optional[ProfilingConfig] = None,
    ):
        self.adapter = adapter
        self.config = config or ProfilingConfig()
        self.sampler = get_sampler(adapter)

    def _classify_pg_type(self, pg_type: str) -> ColumnDataType:
        """Classify PostgreSQL data type."""
        pg_type_lower = pg_type.lower()

        if any(t in pg_type_lower for t in self.PG_NUMERIC_TYPES):
            return ColumnDataType.NUMERIC
        if any(t in pg_type_lower for t in self.PG_TEXT_TYPES):
            return ColumnDataType.TEXT
        if any(t in pg_type_lower for t in self.PG_TIMESTAMP_TYPES):
            return ColumnDataType.TIMESTAMP
        if any(t in pg_type_lower for t in self.PG_BOOLEAN_TYPES):
            return ColumnDataType.BOOLEAN

        return ColumnDataType.UNKNOWN

    async def profile_table(
        self, table: str, database_name: str = ""
    ) -> DataProfile:
        """Generate complete profile for a table."""
        # Get basic info
        row_count = await self.sampler.get_row_count(table)
        column_info = await self.sampler.get_column_info(table)

        # Get schema hash for change detection
        schema_hash = ""
        if isinstance(self.sampler, PostgreSQLSampler):
            schema_hash = await self.sampler.get_schema_hash(table)

        # Profile each column
        column_stats = []
        for col in column_info:
            col_stats = await self._profile_column(
                table, col["name"], col["data_type"]
            )
            column_stats.append(col_stats)

        return DataProfile(
            table=table,
            database=database_name,
            last_updated=datetime.utcnow(),
            row_count=row_count,
            columns=column_stats,
            profile_version=1,
            schema_hash=schema_hash,
        )

    async def _profile_column(
        self, table: str, column: str, raw_type: str
    ) -> ColumnStats:
        """Profile a single column."""
        data_type = self._classify_pg_type(raw_type)

        base_stats = ColumnStats(name=column, data_type=data_type)

        if data_type == ColumnDataType.NUMERIC:
            return await self._profile_numeric_column(table, column, base_stats)
        elif data_type == ColumnDataType.TEXT:
            return await self._profile_text_column(table, column, base_stats)
        else:
            # For timestamp and boolean, just get null ratio
            return await self._profile_basic_column(table, column, base_stats)

    async def _profile_numeric_column(
        self, table: str, column: str, base_stats: ColumnStats
    ) -> ColumnStats:
        """Profile numeric column with distribution detection."""
        if not isinstance(self.sampler, PostgreSQLSampler):
            return base_stats

        # Get basic stats
        stats_dict = await self.sampler.get_numeric_stats(table, column)
        base_stats.min_val = stats_dict.get("min_val")
        base_stats.max_val = stats_dict.get("max_val")
        base_stats.avg_val = stats_dict.get("avg_val")
        base_stats.median_val = stats_dict.get("median_val")
        base_stats.distinct_count = stats_dict.get("distinct_count")
        base_stats.null_ratio = stats_dict.get("null_ratio")

        # Get percentiles if configured
        if self.config.calculate_percentiles:
            percentiles = await self.sampler.get_numeric_percentiles(table, column)
            base_stats.percentiles = percentiles

            # Detect distribution
            if self.config.detect_distribution:
                samples = await self.sampler.sample_column(
                    table, column, self.config.sample_size
                )
                numeric_samples = [
                    float(v) for v in samples if v is not None and isinstance(v, (int, float))
                ]
                if numeric_samples:
                    dist_type = DistributionDetector.detect(numeric_samples, percentiles)
                    base_stats.distribution = Distribution(
                        type=dist_type,
                        p25=percentiles.get("p25"),
                        p75=percentiles.get("p75"),
                        p99=percentiles.get("p99"),
                    )

        return base_stats

    async def _profile_text_column(
        self, table: str, column: str, base_stats: ColumnStats
    ) -> ColumnStats:
        """Profile text column with length stats and top values."""
        if not isinstance(self.sampler, PostgreSQLSampler):
            return base_stats

        # Get text stats
        stats_dict = await self.sampler.get_text_stats(table, column)
        base_stats.distinct_count = stats_dict.get("distinct_count")
        base_stats.null_ratio = stats_dict.get("null_ratio")

        if self.config.analyze_text:
            top_values = await self.sampler.get_top_values(
                table, column, self.config.max_top_values
            )
            base_stats.text_stats = TextStats(
                min_length=stats_dict.get("min_length", 0),
                max_length=stats_dict.get("max_length", 0),
                avg_length=stats_dict.get("avg_length", 0.0),
                empty_ratio=stats_dict.get("empty_ratio", 0.0),
                top_values=top_values,
            )

        return base_stats

    async def _profile_basic_column(
        self, table: str, column: str, base_stats: ColumnStats
    ) -> ColumnStats:
        """Profile column with basic stats only."""
        if isinstance(self.sampler, PostgreSQLSampler):
            text_stats = await self.sampler.get_text_stats(table, column)
            base_stats.distinct_count = text_stats.get("distinct_count")
            base_stats.null_ratio = text_stats.get("null_ratio")
        return base_stats

    async def should_update_profile(
        self, table: str, existing_profile: Optional[DataProfile]
    ) -> bool:
        """Check if profile needs update based on schema changes."""
        if not existing_profile:
            return True

        if not isinstance(self.sampler, PostgreSQLSampler):
            return True

        current_hash = await self.sampler.get_schema_hash(table)
        return current_hash != existing_profile.schema_hash

    @staticmethod
    def compute_schema_hash(column_info: list[dict[str, Any]]) -> str:
        """Compute hash from column info for change detection."""
        content = ",".join(
            f"{c['name']}:{c['data_type']}" for c in column_info
        )
        return hashlib.md5(content.encode()).hexdigest()
