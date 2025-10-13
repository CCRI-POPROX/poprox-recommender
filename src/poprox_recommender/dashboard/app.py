"""FastAPI application exposing a lightweight dashboard for pipeline runs."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from poprox_recommender.persistence import (
    DEFAULT_PERSISTENCE_BUCKET,
    LocalPersistenceManager,
    PersistenceManager,
    get_persistence_manager,
)

from .cloudwatch import BOTO3_AVAILABLE, CloudWatchLogsService, LogEvent
from .service import PipelineDashboardService


@dataclass
class DashboardSettings:
    """Configuration values for the dashboard."""

    backend: str = "auto"
    default_limit: int = 50

    @classmethod
    def from_env(cls) -> "DashboardSettings":
        backend = os.getenv("POPROX_DASHBOARD_BACKEND")
        if backend is None:
            backend = os.getenv("PERSISTENCE_BACKEND", "auto")
        backend = backend.lower()

        default_limit_str = os.getenv("POPROX_DASHBOARD_DEFAULT_LIMIT", "50")
        try:
            default_limit = max(1, min(500, int(default_limit_str)))
        except ValueError:
            default_limit = 50

        return cls(backend=backend, default_limit=default_limit)


def create_app(settings: Optional[DashboardSettings] = None) -> FastAPI:
    """Create a configured FastAPI instance for the dashboard."""

    settings = settings or DashboardSettings.from_env()

    templates = Jinja2Templates(directory=str(_template_dir()))

    app = FastAPI(title="POPROX Pipeline Dashboard")

    static_dir = _static_dir()
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    persistence = _build_persistence(settings)
    service = PipelineDashboardService(persistence)
    app.state.dashboard_service = service
    app.state.dashboard_settings = settings
    app.state.templates = templates
    app.state.data_source_label = _describe_persistence(settings)
    app.state.current_backend = settings.backend

    @app.get("/", response_class=HTMLResponse)
    async def list_sessions_view(
        request: Request,
        request_id: Optional[str] = Query(default=None, description="Filter by request ID"),
        date_param: Optional[str] = Query(default=None, alias="date", description="Filter by YYYY-MM-DD date"),
        limit: int = Query(default=settings.default_limit, ge=1, le=500),
        backend: Optional[str] = Query(default=None, description="Backend to use (local or s3)"),
    ) -> HTMLResponse:
        service: PipelineDashboardService = request.app.state.dashboard_service
        templates: Jinja2Templates = request.app.state.templates

        # Switch backend if requested
        if backend and backend != request.app.state.current_backend:
            persistence = _build_persistence_with_backend(backend)
            service = PipelineDashboardService(persistence)
            request.app.state.dashboard_service = service
            request.app.state.current_backend = backend
            request.app.state.data_source_label = _describe_persistence_backend(backend)

        selected_day = _parse_date_param(date_param)
        summaries = service.list_sessions(limit=limit, request_id=request_id, day=selected_day)

        context = {
            "request": request,
            "sessions": summaries,
            "selected_request_id": request_id or "",
            "selected_date": date_param or "",
            "limit": limit,
            "data_source": request.app.state.data_source_label,
            "current_backend": request.app.state.current_backend,
        }
        return templates.TemplateResponse("index.html", context)

    @app.get("/sessions/{session_id}", response_class=HTMLResponse)
    async def session_detail_view(request: Request, session_id: str) -> HTMLResponse:
        service: PipelineDashboardService = request.app.state.dashboard_service
        templates: Jinja2Templates = request.app.state.templates

        detail = service.get_session(session_id)
        if detail is None:
            raise HTTPException(status_code=404, detail="Session not found")

        context = {
            "request": request,
            "session": detail,
            "data_source": request.app.state.data_source_label,
            "current_backend": request.app.state.current_backend,
        }
        return templates.TemplateResponse("session_detail.html", context)

    @app.get("/api/logs/errors")
    async def fetch_lambda_errors(
        date_param: Optional[str] = Query(default=None, alias="date", description="Filter by YYYY-MM-DD date"),
        limit: int = Query(default=100, ge=1, le=500, description="Maximum number of log events"),
    ) -> JSONResponse:
        """Fetch Lambda execution errors from CloudWatch Logs for a specific date."""
        if not BOTO3_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="CloudWatch Logs integration requires boto3 to be installed",
            )

        # Parse the date parameter
        target_date = _parse_date_param(date_param)
        if target_date is None:
            raise HTTPException(
                status_code=400,
                detail="Invalid or missing date parameter. Use format: YYYY-MM-DD",
            )

        try:
            # Initialize CloudWatch Logs service
            logs_service = CloudWatchLogsService()

            # Fetch errors for the specified day
            log_events = logs_service.fetch_errors_for_day(target_date, limit=limit)

            # Convert to JSON-serializable format
            events_data = [
                {
                    "timestamp": event.timestamp.isoformat(),
                    "message": event.message,
                    "log_stream": event.log_stream,
                    "request_id": event.request_id,
                }
                for event in log_events
            ]

            return JSONResponse(
                content={
                    "success": True,
                    "date": target_date.isoformat(),
                    "count": len(events_data),
                    "events": events_data,
                }
            )

        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error": str(e),
                },
            )

    @app.get("/api/logs/test")
    async def test_cloudwatch_connection() -> JSONResponse:
        """Test if CloudWatch Logs connection is working."""
        if not BOTO3_AVAILABLE:
            return JSONResponse(
                content={
                    "available": False,
                    "error": "boto3 is not installed",
                }
            )

        try:
            logs_service = CloudWatchLogsService()
            connection_ok = logs_service.test_connection()

            return JSONResponse(
                content={
                    "available": True,
                    "connected": connection_ok,
                    "log_group": logs_service.log_group,
                    "region": logs_service.region,
                }
            )
        except Exception as e:
            return JSONResponse(
                content={
                    "available": True,
                    "connected": False,
                    "error": str(e),
                }
            )

    return app


def _template_dir() -> Path:
    return Path(__file__).resolve().parent / "templates"


def _static_dir() -> Path:
    return Path(__file__).resolve().parent / "static"


def _build_persistence(settings: DashboardSettings) -> PersistenceManager:
    backend = settings.backend

    if backend not in ("auto", "s3", "local"):
        raise ValueError(f"Unsupported dashboard backend: {backend}")

    if backend == "s3":
        try:
            from poprox_recommender.persistence.s3 import S3PersistenceManager
        except ImportError as exc:  # pragma: no cover - requires optional dependency
            raise RuntimeError("boto3 is required for S3 persistence") from exc

        bucket = os.getenv("PERSISTENCE_BUCKET", DEFAULT_PERSISTENCE_BUCKET)
        prefix = os.getenv("PERSISTENCE_PREFIX", "pipeline-outputs/")
        return S3PersistenceManager(bucket, prefix)

    if backend == "local":
        persistence_path = os.getenv("PERSISTENCE_PATH", "./data/pipeline_outputs")
        return LocalPersistenceManager(persistence_path)

    return get_persistence_manager()


def _parse_date_param(raw: Optional[str]) -> Optional[date]:
    if raw is None or raw == "":
        return None
    try:
        return datetime.strptime(raw, "%Y-%m-%d").date()
    except ValueError:
        return None


def _describe_persistence(settings: DashboardSettings) -> str:
    backend = settings.backend
    if backend == "s3":
        bucket = os.getenv("PERSISTENCE_BUCKET", DEFAULT_PERSISTENCE_BUCKET)
        prefix = os.getenv("PERSISTENCE_PREFIX", "pipeline-outputs/")
        return f"S3 ({bucket}/{prefix})"
    if backend == "local":
        return f"Local ({os.getenv('PERSISTENCE_PATH', './data/pipeline_outputs')})"
    return "Auto (environment)"


def _describe_persistence_backend(backend: str) -> str:
    """Describe a specific backend by name."""
    if backend == "s3":
        bucket = os.getenv("PERSISTENCE_BUCKET", DEFAULT_PERSISTENCE_BUCKET)
        prefix = os.getenv("PERSISTENCE_PREFIX", "pipeline-outputs/")
        return f"S3 ({bucket}/{prefix})"
    if backend == "local":
        return f"Local ({os.getenv('PERSISTENCE_PATH', './data/pipeline_outputs')})"
    return f"Unknown backend ({backend})"


def _build_persistence_with_backend(backend: str) -> PersistenceManager:
    """Build a persistence manager for a specific backend."""
    if backend not in ("s3", "local"):
        raise ValueError(f"Unsupported backend: {backend}")

    if backend == "s3":
        try:
            from poprox_recommender.persistence.s3 import S3PersistenceManager
        except ImportError as exc:  # pragma: no cover - requires optional dependency
            raise RuntimeError("boto3 is required for S3 persistence") from exc

        bucket = os.getenv("PERSISTENCE_BUCKET", DEFAULT_PERSISTENCE_BUCKET)
        prefix = os.getenv("PERSISTENCE_PREFIX", "pipeline-outputs/")
        return S3PersistenceManager(bucket, prefix)

    persistence_path = os.getenv("PERSISTENCE_PATH", "./data/pipeline_outputs")
    return LocalPersistenceManager(persistence_path)


app = create_app()

__all__ = ["app", "create_app", "DashboardSettings"]
