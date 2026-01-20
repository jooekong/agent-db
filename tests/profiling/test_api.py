"""Tests for profiling API routes."""

from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_db.api.profiling_routes import router, set_profiling_service
from agent_db.metadata.models import (
    ColumnDataType,
    ColumnStats,
    DataProfile,
    Distribution,
    DistributionType,
    ProfilingJob,
    ProfilingJobStatus,
)


@pytest.fixture
def mock_profiling_service():
    """Create mock profiling service."""
    service = MagicMock()

    # Mock profile data
    mock_profile = DataProfile(
        table="users",
        database="postgresql",
        last_updated=datetime(2024, 1, 15, 10, 0, 0),
        row_count=10000,
        columns=[
            ColumnStats(
                name="id",
                data_type=ColumnDataType.NUMERIC,
                min_val=1.0,
                max_val=10000.0,
                distinct_count=10000,
            ),
            ColumnStats(
                name="name",
                data_type=ColumnDataType.TEXT,
                distinct_count=5000,
                null_ratio=0.01,
            ),
        ],
        profile_version=1,
        schema_hash="abc123",
    )

    # Mock job
    mock_job = ProfilingJob(
        job_id="job-123",
        database="postgresql",
        table="users",
        status=ProfilingJobStatus.COMPLETED,
        started_at=datetime(2024, 1, 15, 10, 0, 0),
        completed_at=datetime(2024, 1, 15, 10, 5, 0),
    )

    # Configure mocks
    service.list_profiles.return_value = ["postgresql.users", "postgresql.orders"]
    service.metadata_store.get_profile.return_value = mock_profile
    service.get_profile.return_value = mock_profile
    service.profile_table = AsyncMock(return_value=mock_profile)
    service.profile_database = AsyncMock(return_value=[mock_profile])
    service.list_jobs.return_value = [mock_job]
    service.list_running_jobs.return_value = []
    service.get_job.return_value = mock_job

    return service


@pytest.fixture
def client(mock_profiling_service):
    """Create test client with mock service."""
    app = FastAPI()
    app.include_router(router)
    set_profiling_service(mock_profiling_service)
    return TestClient(app)


class TestProfilingAPI:
    """Tests for profiling API endpoints."""

    def test_list_profiles(self, client, mock_profiling_service):
        """Test GET /profiling/profiles."""
        response = client.get("/profiling/profiles")

        assert response.status_code == 200
        data = response.json()
        assert "profiles" in data
        assert len(data["profiles"]) == 2

    def test_get_profile(self, client, mock_profiling_service):
        """Test GET /profiling/profiles/{database}/{table}."""
        response = client.get("/profiling/profiles/postgresql/users")

        assert response.status_code == 200
        data = response.json()
        assert "profile" in data
        assert data["profile"]["table"] == "users"
        assert data["profile"]["database"] == "postgresql"
        assert data["profile"]["row_count"] == 10000

    def test_get_profile_not_found(self, client, mock_profiling_service):
        """Test GET /profiling/profiles/{database}/{table} for non-existent profile."""
        mock_profiling_service.get_profile.return_value = None

        response = client.get("/profiling/profiles/postgresql/nonexistent")

        assert response.status_code == 404

    def test_create_profile(self, client, mock_profiling_service):
        """Test POST /profiling/profiles/{database}/{table}."""
        response = client.post("/profiling/profiles/postgresql/users")

        assert response.status_code == 200
        data = response.json()
        assert "profile" in data
        mock_profiling_service.profile_table.assert_called_once()

    def test_create_profile_with_force(self, client, mock_profiling_service):
        """Test POST /profiling/profiles/{database}/{table}?force=true."""
        response = client.post("/profiling/profiles/postgresql/users?force=true")

        assert response.status_code == 200
        mock_profiling_service.profile_table.assert_called_once_with(
            "postgresql", "users", force=True
        )

    def test_create_profiles_batch(self, client, mock_profiling_service):
        """Test POST /profiling/profiles (batch)."""
        response = client.post(
            "/profiling/profiles",
            json={
                "database": "postgresql",
                "tables": ["users", "orders"],
                "force": False,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "profiles" in data
        mock_profiling_service.profile_database.assert_called_once()

    def test_create_profile_invalid_database(self, client, mock_profiling_service):
        """Test POST /profiling/profiles with invalid database."""
        mock_profiling_service.profile_table.side_effect = ValueError(
            "No adapter registered"
        )

        response = client.post("/profiling/profiles/invalid/users")

        assert response.status_code == 400

    def test_list_jobs(self, client, mock_profiling_service):
        """Test GET /profiling/jobs."""
        response = client.get("/profiling/jobs")

        assert response.status_code == 200
        data = response.json()
        assert "jobs" in data
        assert len(data["jobs"]) == 1
        assert data["jobs"][0]["job_id"] == "job-123"

    def test_list_running_jobs(self, client, mock_profiling_service):
        """Test GET /profiling/jobs/running."""
        response = client.get("/profiling/jobs/running")

        assert response.status_code == 200
        data = response.json()
        assert "jobs" in data
        assert len(data["jobs"]) == 0

    def test_get_job(self, client, mock_profiling_service):
        """Test GET /profiling/jobs/{job_id}."""
        response = client.get("/profiling/jobs/job-123")

        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == "job-123"
        assert data["status"] == "completed"

    def test_get_job_not_found(self, client, mock_profiling_service):
        """Test GET /profiling/jobs/{job_id} for non-existent job."""
        mock_profiling_service.get_job.return_value = None

        response = client.get("/profiling/jobs/nonexistent")

        assert response.status_code == 404


class TestProfilingServiceNotConfigured:
    """Test behavior when profiling service is not configured."""

    def test_list_profiles_service_not_configured(self):
        """Test that 503 is returned when service not configured."""
        app = FastAPI()
        app.include_router(router)
        set_profiling_service(None)
        client = TestClient(app)

        response = client.get("/profiling/profiles")

        assert response.status_code == 503
        assert "not configured" in response.json()["detail"]
