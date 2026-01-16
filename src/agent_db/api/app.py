"""FastAPI application factory."""

from fastapi import FastAPI

from agent_db.api.routes import router


def create_app() -> FastAPI:
    """Create FastAPI application."""
    app = FastAPI(
        title="Agent-DB",
        description="AI-Native Data Interaction Platform",
        version="0.1.0",
    )

    app.include_router(router)

    return app
