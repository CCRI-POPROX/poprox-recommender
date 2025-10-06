"""Utilities for reading persisted pipeline data for the dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, Iterable, List, Optional

from poprox_recommender.persistence import PersistenceManager


@dataclass
class SessionSummary:
    """High-level view of a persisted pipeline session."""

    session_id: str
    request_id: str
    timestamp: datetime
    num_articles: int
    issue_count: int
    component_summary: Dict[str, Dict[str, Any]]
    storage_location: Optional[str]
    llm_summary: Optional[Dict[str, Any]]
    pipeline: Optional[str]
    timeout_info: Optional[Dict[str, Any]]

    @property
    def day(self) -> date:
        return self.timestamp.date()


@dataclass
class ArticleView:
    """Renderable view of an article before/after rewriting."""

    position: int
    article_id: Optional[str]
    original_headline: str
    rewritten_headline: str
    subhead: Optional[str]
    rewrite_status: str
    metrics: Dict[str, Any]


@dataclass
class SessionDetail:
    """Full detail for a persisted pipeline session."""

    session_id: str
    request_id: str
    timestamp: datetime
    user_model: str
    metadata: Dict[str, Any]
    articles: List[ArticleView]


class PipelineDashboardService:
    """Helper that adapts persistence managers for dashboard consumption."""

    def __init__(self, persistence: PersistenceManager):
        self._persistence = persistence

    def list_sessions(
        self,
        *,
        limit: int = 50,
        request_id: Optional[str] = None,
        day: Optional[date] = None,
    ) -> List[SessionSummary]:
        """List recent sessions filtered by optional request ID and day."""
        summaries: List[SessionSummary] = []
        request_prefix = request_id if request_id else None

        session_ids = self._persistence.list_sessions(request_prefix)

        for session_id in session_ids:
            if len(summaries) >= limit:
                break

            try:
                metadata = self._persistence.load_metadata(session_id)
            except Exception:
                # Skip sessions we cannot read metadata for
                continue

            timestamp = _parse_timestamp(metadata.get("timestamp"))
            if timestamp is None:
                continue

            if day and timestamp.date() != day:
                continue

            metadata_request_id = metadata.get("request_id", "")
            if request_id and metadata_request_id != request_id:
                continue

            num_articles = metadata.get("num_articles", 0)
            issue_count = metadata.get("issue_count", 0)
            component_summary = metadata.get("component_summary", {})
            storage_location = metadata.get("storage_location")
            llm_summary = metadata.get("llm_summary")
            pipeline = metadata.get("pipeline_type")
            timeout_info = metadata.get("timeout_info")

            summaries.append(
                SessionSummary(
                    session_id=session_id,
                    request_id=metadata_request_id,
                    timestamp=timestamp,
                    num_articles=num_articles,
                    issue_count=issue_count,
                    component_summary=component_summary,
                    storage_location=storage_location,
                    llm_summary=llm_summary,
                    pipeline=pipeline,
                    timeout_info=timeout_info,
                )
            )

        return summaries

    def get_session(self, session_id: str) -> Optional[SessionDetail]:
        """Return full session detail or ``None`` if unavailable."""
        try:
            payload = self._persistence.load_pipeline_data(session_id)
        except Exception:
            return None

        metadata = payload.get("metadata", {})
        timestamp = _parse_timestamp(metadata.get("timestamp"))
        if timestamp is None:
            return None

        request_id = metadata.get("request_id", "")
        user_model = payload.get("user_model", "")

        original = getattr(payload.get("original_recommendations"), "articles", [])
        rewritten = getattr(payload.get("rewritten_recommendations"), "articles", [])

        rewriter_metrics = (
            metadata.get("llm_metrics", {}).get("rewriter", []) if metadata.get("llm_metrics") else []
        )
        metrics_by_article = {
            str(metric.get("article_id")): metric for metric in rewriter_metrics if metric.get("article_id") is not None
        }

        articles: List[ArticleView] = []
        for idx, (orig, rew) in enumerate(_zip_strict(original, rewritten)):
            article_id = _safe_str(getattr(orig, "article_id", getattr(rew, "article_id", idx)))
            metrics = metrics_by_article.get(article_id, {})
            articles.append(
                ArticleView(
                    position=idx + 1,
                    article_id=article_id,
                    original_headline=_safe_str(getattr(orig, "headline", "")),
                    rewritten_headline=_safe_str(getattr(rew, "headline", "")),
                    subhead=_safe_optional_str(getattr(orig, "subhead", None)),
                    rewrite_status=metrics.get("status", "unknown"),
                    metrics=metrics,
                )
            )

        return SessionDetail(
            session_id=session_id,
            request_id=request_id,
            timestamp=timestamp,
            user_model=user_model,
            metadata=metadata,
            articles=articles,
        )


def _parse_timestamp(raw: Any) -> Optional[datetime]:
    if not raw or not isinstance(raw, str):
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def _safe_str(value: Any) -> str:
    return "" if value is None else str(value)


def _safe_optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(value)


def _zip_strict(left: Iterable[Any], right: Iterable[Any]) -> Iterable[tuple[Any, Any]]:
    left_list = list(left)
    right_list = list(right)
    length = min(len(left_list), len(right_list))
    return zip(left_list[:length], right_list[:length])
