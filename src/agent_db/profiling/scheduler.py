"""Scheduler for periodic profiling tasks."""

import asyncio
from datetime import datetime
from typing import Any, Callable, Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from agent_db.metadata.models import ProfilingConfig
from agent_db.profiling.service import ProfilingService


class ProfilingScheduler:
    """Scheduler for periodic data profiling."""

    def __init__(
        self,
        profiling_service: ProfilingService,
        config: Optional[ProfilingConfig] = None,
    ):
        self.service = profiling_service
        self.config = config or ProfilingConfig()
        self._scheduler: Optional[AsyncIOScheduler] = None
        self._is_running = False

    def start(self) -> None:
        """Start the scheduler."""
        if self._is_running:
            return

        self._scheduler = AsyncIOScheduler()

        # Add default daily job if cron configured
        if self.config.schedule_cron:
            self.add_cron_job(
                "default_profiling",
                self._run_all_profiles,
                self.config.schedule_cron,
            )

        self._scheduler.start()
        self._is_running = True

    def stop(self) -> None:
        """Stop the scheduler."""
        if self._scheduler:
            self._scheduler.shutdown(wait=False)
            self._scheduler = None
        self._is_running = False

    def add_cron_job(
        self,
        job_id: str,
        func: Callable,
        cron_expression: str,
        **kwargs: Any,
    ) -> None:
        """Add a cron-scheduled job."""
        if not self._scheduler:
            raise RuntimeError("Scheduler not started")

        # Parse cron expression (minute hour day month day_of_week)
        parts = cron_expression.split()
        if len(parts) == 5:
            trigger = CronTrigger(
                minute=parts[0],
                hour=parts[1],
                day=parts[2],
                month=parts[3],
                day_of_week=parts[4],
            )
        else:
            raise ValueError(f"Invalid cron expression: {cron_expression}")

        self._scheduler.add_job(
            func,
            trigger=trigger,
            id=job_id,
            replace_existing=True,
            **kwargs,
        )

    def add_interval_job(
        self,
        job_id: str,
        func: Callable,
        hours: int = 0,
        minutes: int = 0,
        seconds: int = 0,
        **kwargs: Any,
    ) -> None:
        """Add an interval-scheduled job."""
        if not self._scheduler:
            raise RuntimeError("Scheduler not started")

        self._scheduler.add_job(
            func,
            "interval",
            hours=hours,
            minutes=minutes,
            seconds=seconds,
            id=job_id,
            replace_existing=True,
            **kwargs,
        )

    def remove_job(self, job_id: str) -> None:
        """Remove a scheduled job."""
        if self._scheduler:
            try:
                self._scheduler.remove_job(job_id)
            except Exception:
                pass

    def get_jobs(self) -> list[dict[str, Any]]:
        """Get all scheduled jobs."""
        if not self._scheduler:
            return []

        jobs = []
        for job in self._scheduler.get_jobs():
            jobs.append({
                "id": job.id,
                "name": job.name,
                "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
                "trigger": str(job.trigger),
            })
        return jobs

    async def _run_all_profiles(self) -> None:
        """Run profiling for all registered databases."""
        for db_name in self.service.get_adapters().keys():
            try:
                await self.service.profile_database(db_name)
            except Exception:
                # Log error but continue with other databases
                pass

    async def run_profile_job(
        self,
        database: str,
        tables: Optional[list[str]] = None,
    ) -> None:
        """Manually trigger a profiling job."""
        await self.service.profile_database(database, tables=tables)

    @property
    def is_running(self) -> bool:
        """Check if scheduler is running."""
        return self._is_running
