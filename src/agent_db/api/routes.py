"""API routes."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from agent_db.engine.interpreter import InterpretedResult

router = APIRouter()


class QueryRequest(BaseModel):
    """Query request body."""

    question: str = Field(..., min_length=1)


class QueryResponse(BaseModel):
    """Query response body."""

    summary: str
    suggestions: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str


# Global service instance (set during app startup)
_query_service = None


def set_query_service(service):
    """Set the global query service instance."""
    global _query_service
    _query_service = service


async def get_query_service():
    """Get query service instance."""
    if _query_service is None:
        raise HTTPException(status_code=503, detail="Query service not configured")
    return _query_service


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="healthy", version="0.1.0")


@router.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    service=Depends(get_query_service),
):
    """Execute natural language query."""
    try:
        result: InterpretedResult = await service.query(request.question)
        return QueryResponse(
            summary=result.summary,
            suggestions=result.suggestions,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
