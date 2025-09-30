"""Command-line entry point for the dashboard."""

from __future__ import annotations

import os

import uvicorn

from .app import DashboardSettings, create_app


def main() -> None:
    settings = DashboardSettings.from_env()
    app = create_app(settings)

    host = os.getenv("POPROX_DASHBOARD_HOST", "127.0.0.1")
    port = int(os.getenv("POPROX_DASHBOARD_PORT", "8000"))
    reload_flag = os.getenv("POPROX_DASHBOARD_RELOAD", "false").lower() in {"1", "true", "yes"}

    uvicorn.run(app, host=host, port=port, reload=reload_flag, log_level="info")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
