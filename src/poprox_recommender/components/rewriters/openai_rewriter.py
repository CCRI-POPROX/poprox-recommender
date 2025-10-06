import asyncio
import copy
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

import openai
from dotenv import load_dotenv
from lenskit.pipeline import Component
from pydantic import BaseModel, Field

from poprox_concepts.domain import RecommendationList
from poprox_recommender.persistence import get_persistence_manager

load_dotenv()


class LlmResponse(BaseModel):
    headline: str


class LLMRewriterConfig(BaseModel):
    """
    Configuration for the LLM-powered rewriter.
    """

    model: str = "gpt-4.1-2025-04-14"
    openai_api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    pipeline_name: str = Field(default="llm_rank_rewrite", description="Name of the pipeline using this rewriter")


class LLMRewriter(Component):
    config: LLMRewriterConfig

    def __call__(self, ranker_output: tuple[RecommendationList, str, str, dict, dict]) -> RecommendationList:
        (
            recommendations,
            user_model,
            request_id,
            ranker_metrics,
            ranker_component_metrics,
        ) = ranker_output

        # Create a deep copy of the original recommendations for persistence
        original_recommendations = RecommendationList(articles=copy.deepcopy(recommendations.articles))

        with open("prompts/rewrite.md", "r") as f:
            prompt = f.read()

        rewriter_metrics: List[Dict[str, Any]] = []
        component_issues: List[Dict[str, Any]] = []

        component_meta: Dict[str, Any] = {
            "component": "rewriter",
            "start_time": datetime.now(timezone.utc).isoformat(),
            "status": "in_progress",
        }
        component_start = time.perf_counter()
        component_snapshot: Dict[str, Any] | None = None

        def finalize_component(
            status: str,
            exc: Exception | None = None,
            *,
            error_count: int | None = None,
        ) -> Dict[str, Any]:
            component_meta["status"] = status
            if exc is not None:
                component_meta["error_type"] = type(exc).__name__
                component_meta["error_message"] = str(exc)
            if error_count is not None:
                component_meta["error_count"] = error_count
            component_meta["end_time"] = datetime.now(timezone.utc).isoformat()
            component_meta["duration_seconds"] = time.perf_counter() - component_start
            return component_meta.copy()

        # rewrite article headlines in parallel
        async def rewrite_article(art, client):
            # build prompt using the user_model from LLMRanker
            input_txt = f"""User interest profile:
{user_model}

Headline to rewrite:
{art.headline}

Article text:
{art.body}
"""

            metrics: Dict[str, Any] = {
                "article_id": str(art.article_id),
                "original_headline": art.headline,
                "status": "in_progress",
                "start_time": datetime.now(timezone.utc).isoformat(),
            }
            call_start = time.perf_counter()

            try:
                response = await client.responses.parse(
                    model=self.config.model,
                    instructions=prompt,
                    input=input_txt,
                    temperature=0.5,
                    text_format=LlmResponse,
                )
                metrics.update(
                    {
                        "status": "success",
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens,
                        "duration_seconds": time.perf_counter() - call_start,
                    }
                )
                # Update the article headline with the rewritten one
                art.headline = response.output_parsed.headline
                metrics["rewritten_headline"] = art.headline
            except Exception as exc:  # pragma: no cover - defensive logging for production observability
                metrics.update(
                    {
                        "status": "error",
                        "duration_seconds": time.perf_counter() - call_start,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                )
                component_issues.append(
                    {
                        "component": "rewriter",
                        "context": {
                            "article_id": metrics["article_id"],
                            "headline": metrics.get("original_headline"),
                        },
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                )
                logging.error("Failed to rewrite article %s: %s", metrics["article_id"], exc, exc_info=exc)
            finally:
                metrics["end_time"] = datetime.now(timezone.utc).isoformat()
                if "duration_seconds" not in metrics:
                    metrics["duration_seconds"] = time.perf_counter() - call_start
                rewriter_metrics.append(metrics)

        async def rewrite_all():
            async with openai.AsyncOpenAI(api_key=self.config.openai_api_key) as client:
                tasks = [rewrite_article(art, client) for art in recommendations.articles]
                await asyncio.gather(*tasks)

        try:
            asyncio.run(rewrite_all())
        except Exception as exc:  # pragma: no cover - defensive logging for production observability
            component_issues.append(
                {
                    "component": "rewriter",
                    "context": {"stage": "rewrite_all"},
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
            component_snapshot = finalize_component("error", exc, error_count=len(component_issues))
            raise
        else:
            article_errors = [m for m in rewriter_metrics if m.get("status") != "success"]
            if article_errors:
                component_snapshot = finalize_component(
                    "partial_failure",
                    error_count=len(article_errors),
                )
            else:
                component_snapshot = finalize_component("success", error_count=0)

        # Persist pipeline data after rewriting is complete
        try:
            persistence = get_persistence_manager()

            if component_snapshot is None:  # pragma: no cover - defensive fallback
                component_snapshot = component_meta.copy()

            # Combine all metrics
            combined_metrics = {
                "pipeline_type": self.config.pipeline_name,
                "rewriter_model": self.config.model,
                "llm_metrics": {
                    "ranker": ranker_metrics,
                    "rewriter": rewriter_metrics,
                },
                "component_metrics": {
                    **(ranker_component_metrics or {}),
                    "rewriter": component_snapshot,
                },
                "issues": component_issues,
            }

            session_id = persistence.save_pipeline_data(
                request_id=request_id,
                user_model=user_model,
                original_recommendations=original_recommendations,
                rewritten_recommendations=recommendations,
                metadata=combined_metrics,
            )
            logging.info(f"Pipeline data saved with session_id: {session_id}")
        except Exception as e:
            # Log the error but don't fail the pipeline
            logging.error(f"Failed to persist pipeline data: {e}")

        return recommendations
