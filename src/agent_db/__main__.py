"""CLI entry point."""

import uvicorn

from agent_db.api.app import create_app


def main():
    """Run the API server."""
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
