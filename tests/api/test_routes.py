"""Tests for API routes."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from agent_db.api.app import create_app
from agent_db.api.routes import set_query_service
from agent_db.engine.interpreter import InterpretedResult


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_check(self, client: TestClient):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


class TestQueryEndpoint:
    def test_query_endpoint_no_service(self, client: TestClient):
        response = client.post(
            "/query",
            json={"question": "How many users?"},
        )
        assert response.status_code == 503

    def test_query_endpoint_with_service(self, client: TestClient):
        mock_service = MagicMock()
        mock_service.query = AsyncMock(return_value=InterpretedResult(
            summary="You have 1000 users.",
            suggestions=["Check user growth trend"],
        ))
        set_query_service(mock_service)

        response = client.post(
            "/query",
            json={"question": "How many users?"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "summary" in data
        assert "1000" in data["summary"]

        # Cleanup
        set_query_service(None)

    def test_query_endpoint_validation(self, client: TestClient):
        # Need to set service first, otherwise 503 is returned before validation
        mock_service = MagicMock()
        set_query_service(mock_service)

        response = client.post("/query", json={})
        assert response.status_code == 422

        # Cleanup
        set_query_service(None)
