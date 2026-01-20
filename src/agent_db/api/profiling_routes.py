"""API routes for data profiling."""

from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from agent_db.metadata.models import DataProfile, ProfilingJob, ProfilingJobStatus


router = APIRouter(prefix="/profiling", tags=["profiling"])


# Request/Response models
class ProfileRequest(BaseModel):
    """Request to trigger profiling."""

    database: str
    tables: Optional[list[str]] = None
    force: bool = False


class ProfileResponse(BaseModel):
    """Profile response."""

    table: str
    database: str
    row_count: int
    column_count: int
    last_updated: str


class ProfileDetailResponse(BaseModel):
    """Detailed profile response."""

    profile: dict[str, Any]


class JobResponse(BaseModel):
    """Job response."""

    job_id: str
    database: str
    table: str
    status: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None


class JobListResponse(BaseModel):
    """List of jobs response."""

    jobs: list[JobResponse]


class ProfileListResponse(BaseModel):
    """List of profiles response."""

    profiles: list[ProfileResponse]


# Global profiling service instance
_profiling_service = None


def set_profiling_service(service) -> None:
    """Set the global profiling service instance."""
    global _profiling_service
    _profiling_service = service


async def get_profiling_service():
    """Get profiling service instance."""
    if _profiling_service is None:
        raise HTTPException(status_code=503, detail="Profiling service not configured")
    return _profiling_service


@router.get("/profiles", response_model=ProfileListResponse)
async def list_profiles(service=Depends(get_profiling_service)):
    """List all available data profiles."""
    profile_names = service.list_profiles()
    profiles = []

    for name in profile_names:
        profile = service.metadata_store.get_profile(name)
        if profile:
            profiles.append(ProfileResponse(
                table=profile.table,
                database=profile.database,
                row_count=profile.row_count,
                column_count=len(profile.columns),
                last_updated=profile.last_updated.isoformat(),
            ))

    return ProfileListResponse(profiles=profiles)


@router.get("/profiles/{database}/{table}", response_model=ProfileDetailResponse)
async def get_profile(
    database: str,
    table: str,
    service=Depends(get_profiling_service),
):
    """Get detailed profile for a specific table."""
    profile = service.get_profile(database, table)
    if not profile:
        raise HTTPException(status_code=404, detail=f"Profile not found: {database}.{table}")

    return ProfileDetailResponse(profile=profile.model_dump(mode="json"))


@router.post("/profiles/{database}/{table}", response_model=ProfileDetailResponse)
async def create_profile(
    database: str,
    table: str,
    force: bool = False,
    service=Depends(get_profiling_service),
):
    """Trigger profiling for a specific table."""
    try:
        profile = await service.profile_table(database, table, force=force)
        return ProfileDetailResponse(profile=profile.model_dump(mode="json"))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/profiles", response_model=ProfileListResponse)
async def create_profiles(
    request: ProfileRequest,
    service=Depends(get_profiling_service),
):
    """Trigger profiling for multiple tables."""
    try:
        profiles = await service.profile_database(
            request.database,
            tables=request.tables,
            force=request.force,
        )

        response_profiles = [
            ProfileResponse(
                table=p.table,
                database=p.database,
                row_count=p.row_count,
                column_count=len(p.columns),
                last_updated=p.last_updated.isoformat(),
            )
            for p in profiles
        ]

        return ProfileListResponse(profiles=response_profiles)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs", response_model=JobListResponse)
async def list_jobs(service=Depends(get_profiling_service)):
    """List all profiling jobs."""
    jobs = service.list_jobs()

    return JobListResponse(jobs=[
        JobResponse(
            job_id=j.job_id,
            database=j.database,
            table=j.table,
            status=j.status.value,
            started_at=j.started_at.isoformat() if j.started_at else None,
            completed_at=j.completed_at.isoformat() if j.completed_at else None,
            error_message=j.error_message,
        )
        for j in jobs
    ])


@router.get("/jobs/running", response_model=JobListResponse)
async def list_running_jobs(service=Depends(get_profiling_service)):
    """List running profiling jobs."""
    jobs = service.list_running_jobs()

    return JobListResponse(jobs=[
        JobResponse(
            job_id=j.job_id,
            database=j.database,
            table=j.table,
            status=j.status.value,
            started_at=j.started_at.isoformat() if j.started_at else None,
            completed_at=None,
            error_message=None,
        )
        for j in jobs
    ])


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str, service=Depends(get_profiling_service)):
    """Get job details by ID."""
    job = service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    return JobResponse(
        job_id=job.job_id,
        database=job.database,
        table=job.table,
        status=job.status.value,
        started_at=job.started_at.isoformat() if job.started_at else None,
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
        error_message=job.error_message,
    )
