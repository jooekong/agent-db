"""Profiling service for orchestrating data profile generation."""

import asyncio
import uuid
from datetime import datetime
from typing import Any, Optional

from agent_db.adapters.protocol import DatabaseAdapter
from agent_db.metadata.models import (
    DataProfile,
    ProfilingConfig,
    ProfilingJob,
    ProfilingJobStatus,
)
from agent_db.metadata.store import MetadataStore
from agent_db.profiling.engine import ProfilerEngine


class ProfilingService:
    """Service for managing data profiling operations."""

    def __init__(
        self,
        metadata_store: MetadataStore,
        config: Optional[ProfilingConfig] = None,
    ):
        self.metadata_store = metadata_store
        self.config = config or ProfilingConfig()
        self._adapters: dict[str, DatabaseAdapter] = {}
        self._engines: dict[str, ProfilerEngine] = {}
        self._jobs: dict[str, ProfilingJob] = {}

    def add_adapter(self, name: str, adapter: DatabaseAdapter) -> None:
        """Register a database adapter."""
        self._adapters[name] = adapter
        self._engines[name] = ProfilerEngine(adapter, self.config)

    def remove_adapter(self, name: str) -> None:
        """Remove a database adapter."""
        self._adapters.pop(name, None)
        self._engines.pop(name, None)

    def get_adapters(self) -> dict[str, DatabaseAdapter]:
        """Get all registered adapters."""
        return self._adapters.copy()

    async def profile_table(
        self,
        database: str,
        table: str,
        force: bool = False,
    ) -> DataProfile:
        """Profile a single table."""
        engine = self._engines.get(database)
        if not engine:
            raise ValueError(f"No adapter registered for database: {database}")

        # Check if update needed
        existing = self.metadata_store.get_profile(f"{database}.{table}")
        if not force and existing:
            should_update = await engine.should_update_profile(table, existing)
            if not should_update:
                return existing

        # Generate new profile
        profile = await engine.profile_table(table, database)

        # Save to store
        self.metadata_store.save_profile(profile)

        return profile

    async def profile_database(
        self,
        database: str,
        tables: Optional[list[str]] = None,
        force: bool = False,
    ) -> list[DataProfile]:
        """Profile multiple tables in a database."""
        engine = self._engines.get(database)
        if not engine:
            raise ValueError(f"No adapter registered for database: {database}")

        # If no tables specified, profile all (requires adapter support)
        if tables is None:
            tables = []  # Would need to query information_schema

        profiles = []
        for table in tables:
            try:
                profile = await self.profile_table(database, table, force=force)
                profiles.append(profile)
            except Exception as e:
                # Log error but continue with other tables
                pass

        return profiles

    def get_profile(self, database: str, table: str) -> Optional[DataProfile]:
        """Get profile for a table."""
        return self.metadata_store.get_profile(f"{database}.{table}")

    def list_profiles(self) -> list[str]:
        """List all available profiles."""
        return self.metadata_store.list_profiles()

    def get_all_profiles(self) -> list[DataProfile]:
        """Get all available profiles."""
        profiles = []
        for name in self.list_profiles():
            profile = self.metadata_store.get_profile(name)
            if profile:
                profiles.append(profile)
        return profiles

    # Job management
    def create_job(self, database: str, table: str) -> ProfilingJob:
        """Create a new profiling job."""
        job = ProfilingJob(
            job_id=str(uuid.uuid4()),
            database=database,
            table=table,
            status=ProfilingJobStatus.PENDING,
        )
        self._jobs[job.job_id] = job
        return job

    def get_job(self, job_id: str) -> Optional[ProfilingJob]:
        """Get job by ID."""
        return self._jobs.get(job_id)

    def list_jobs(self) -> list[ProfilingJob]:
        """List all jobs."""
        return list(self._jobs.values())

    def list_running_jobs(self) -> list[ProfilingJob]:
        """List running jobs."""
        return [j for j in self._jobs.values() if j.status == ProfilingJobStatus.RUNNING]

    async def run_job(self, job_id: str) -> DataProfile:
        """Run a profiling job."""
        job = self._jobs.get(job_id)
        if not job:
            raise ValueError(f"Job not found: {job_id}")

        job.status = ProfilingJobStatus.RUNNING
        job.started_at = datetime.utcnow()

        try:
            profile = await self.profile_table(job.database, job.table, force=True)
            job.status = ProfilingJobStatus.COMPLETED
            job.completed_at = datetime.utcnow()
            return profile
        except Exception as e:
            job.status = ProfilingJobStatus.FAILED
            job.completed_at = datetime.utcnow()
            job.error_message = str(e)
            raise

    async def run_job_async(self, job_id: str) -> None:
        """Run job in background."""
        await self.run_job(job_id)

    def format_profiles_for_llm(
        self, databases: Optional[list[str]] = None
    ) -> str:
        """Format profiles as context for LLM."""
        profiles = self.get_all_profiles()
        if databases:
            profiles = [p for p in profiles if p.database in databases]

        lines = []
        for profile in profiles:
            lines.append(f"## {profile.table}")
            lines.append(f"  Row count: {profile.row_count:,}")

            for col in profile.columns:
                parts = [f"  - {col.name}: type={col.data_type.value}"]

                if col.distinct_count is not None:
                    parts.append(f"distinct={col.distinct_count:,}")

                if col.null_ratio is not None and col.null_ratio > 0:
                    parts.append(f"null_ratio={col.null_ratio:.1%}")

                if col.distribution:
                    parts.append(f"distribution={col.distribution.type.value}")

                if col.min_val is not None and col.max_val is not None:
                    parts.append(f"range=[{col.min_val}, {col.max_val}]")

                lines.append(", ".join(parts))

            lines.append("")

        return "\n".join(lines)
