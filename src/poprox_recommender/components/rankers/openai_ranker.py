import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Union

import openai
from dotenv import load_dotenv
from lenskit.pipeline import Component
from pydantic import BaseModel, Field

from poprox_concepts import CandidateSet, InterestProfile
from poprox_concepts.domain import RecommendationList

load_dotenv()


class LlmResponse(BaseModel):
    """
    Structured output: list of selected articles.
    """

    recommended_article_ids: list[int]
    explanation: str


class LLMRankerConfig(BaseModel):
    """
    Configuration for the LLM-powered ranker.
    """

    model: str = "gpt-4.1-mini-2025-04-14"
    num_slots: int = 10
    openai_api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))


class RankedIndices(BaseModel):
    """
    Structured output: list of selected indices from the candidate pool.
    """

    indices: list[int]


class LLMRanker(Component):
    config: LLMRankerConfig

    def __init__(self, config: LLMRankerConfig):
        self.config = config
        self.llm_metrics = {}
        self.component_metrics: Dict[str, dict] = {}

    def _structure_interest_profile(
        self, interest_profile: InterestProfile, articles_clicked: Union[CandidateSet, None]
    ) -> str:
        """
        Structure the interest profile for the prompt.
        """
        click_history = getattr(articles_clicked, "articles", [])
        if click_history:
            clicked_stories = sorted(
                [(art.headline, art.published_at) for art in click_history], key=lambda x: x[1], reverse=True
            )
            clicked_headlines = [headline for headline, _ in clicked_stories]
        else:
            clicked_headlines = []

        clean_profile = {
            "topics": [
                t.entity_name
                for t in sorted(
                    interest_profile.onboarding_topics, key=lambda t: getattr(t, "preference", 0), reverse=True
                )
            ],
            "click_topic_counts": getattr(interest_profile, "click_topic_counts", None),
            "click_locality_counts": getattr(interest_profile, "click_locality_counts", None),
            "click_history": clicked_headlines,
        }
        # Sort the click counts from most clicked to least clicked
        clean_profile["click_topic_counts"] = (
            [t for t, _ in sorted(clean_profile["click_topic_counts"].items(), key=lambda x: x[1], reverse=True)]
            if clean_profile["click_topic_counts"]
            else []
        )
        clean_profile["click_locality_counts"] = (
            [t for t, _ in sorted(clean_profile["click_locality_counts"].items(), key=lambda x: x[1], reverse=True)]
            if clean_profile["click_locality_counts"]
            else []
        )

        profile_str = f"""Topics the user has shown interest in (from most to least):
{", ".join(clean_profile["topics"])}

Topics the user has clicked on (from most to least):
{", ".join(clean_profile["click_topic_counts"])}

Localities the user has clicked on (from most to least):
{", ".join(clean_profile["click_locality_counts"])}

Headlines of articles the user has clicked on (most recent first):
{", ".join(cleaned_headline for cleaned_headline in clean_profile["click_history"] if cleaned_headline)}
"""

        return profile_str

    def _build_user_model(self, interest_profile_str: str) -> str:
        """
        Build a concise user model from the provided interest data.
        """
        with open("prompts/user_profile.md", "r") as f:
            prompt = f.read()

        client = openai.OpenAI(api_key=self.config.openai_api_key)

        start_time = time.time()
        response = client.responses.create(
            model="gpt-4.1-mini-2025-04-14",
            instructions=prompt,
            input=interest_profile_str,
        )
        end_time = time.time()

        self.llm_metrics["user_profile_generation"] = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "duration_seconds": end_time - start_time,
        }

        return response.output_text

    def __call__(
        self,
        candidate_articles: CandidateSet,
        interest_profile: InterestProfile,
        articles_clicked: Optional[CandidateSet] = None,
    ) -> tuple[RecommendationList, str, str, dict, dict]:
        # Reset metrics for this invocation
        self.llm_metrics = {}
        self.component_metrics = {}

        component_meta: Dict[str, Any] = {
            "component": "ranker",
            "start_time": datetime.now(timezone.utc).isoformat(),
            "status": "in_progress",
        }
        component_start = time.perf_counter()

        def finalize_component(status: str, exc: Exception | None = None, *, error_count: int | None = None) -> Dict[str, Any]:
            component_meta["status"] = status
            if exc is not None:
                component_meta["error_type"] = type(exc).__name__
                component_meta["error_message"] = str(exc)
            if error_count is not None:
                component_meta["error_count"] = error_count
            elif status == "success":
                component_meta["error_count"] = 0
            elif status == "error":
                component_meta.setdefault("error_count", 1)
            component_meta["end_time"] = datetime.now(timezone.utc).isoformat()
            component_meta["duration_seconds"] = time.perf_counter() - component_start
            self.component_metrics["ranker"] = component_meta
            return component_meta.copy()
        # Generate a request ID for this pipeline run
        self.request_id = str(interest_profile.profile_id)

        with open("prompts/rank.md", "r") as f:
            prompt = f.read()

        client = openai.OpenAI(api_key=self.config.openai_api_key)

        # build concise profile for prompt
        try:
            profile_str = self._structure_interest_profile(interest_profile, articles_clicked)
            # build user model
            user_model = self._build_user_model(profile_str)

            # summarize candidates for the prompt
            items = []
            for i, art in enumerate(candidate_articles.articles):
                items.append(f"{i}: {art.headline} - {art.subhead}")
            component_meta["num_candidates"] = len(candidate_articles.articles)

            input_txt = f"""User interest profile:
{user_model}

Candidate articles:
{", ".join(items)}

Make sure you select EXACTLY {self.config.num_slots} articles from the candidate pool.
"""
            start_time = time.time()
            response = client.responses.parse(
                model=self.config.model, instructions=prompt, input=input_txt, text_format=LlmResponse, temperature=0.5
            )
            end_time = time.time()

            self.llm_metrics["article_ranking"] = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "duration_seconds": end_time - start_time,
            }

            parsed_response = response.output_parsed
            selected = [candidate_articles.articles[i] for i in parsed_response.recommended_article_ids]
            original_recommendations = RecommendationList(articles=selected)

            metrics_snapshot = finalize_component("success")
            return (
                original_recommendations,
                user_model,
                self.request_id,
                self.llm_metrics,
                {"ranker": metrics_snapshot},
            )
        except Exception as exc:  # pragma: no cover - defensive logging for production observability
            finalize_component("error", exc, error_count=1)
            raise
