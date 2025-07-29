import asyncio
import copy
import logging
import os
import time

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


class LLMRewriter(Component):
    config: LLMRewriterConfig

    def __call__(self, ranker_output: tuple[RecommendationList, str, str, dict]) -> RecommendationList:
        recommendations, user_model, request_id, ranker_metrics = ranker_output

        # Create a deep copy of the original recommendations for persistence
        original_recommendations = RecommendationList(articles=copy.deepcopy(recommendations.articles))

        with open("prompts/rewrite.md", "r") as f:
            prompt = f.read()

        rewriter_metrics = []

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

            start_time = time.time()
            response = await client.responses.parse(
                model=self.config.model,
                instructions=prompt,
                input=input_txt,
                temperature=0.5,
                text_format=LlmResponse,
            )
            end_time = time.time()

            rewriter_metrics.append({
                "article_id": str(art.article_id),
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "duration_seconds": end_time - start_time,
            })

            # Update the article headline with the rewritten one
            art.headline = response.output_parsed.headline

        async def rewrite_all():
            async with openai.AsyncOpenAI(api_key=self.config.openai_api_key) as client:
                tasks = [rewrite_article(art, client) for art in recommendations.articles]
                await asyncio.gather(*tasks)

        asyncio.run(rewrite_all())

        # Persist pipeline data after rewriting is complete
        try:
            persistence = get_persistence_manager()

            # Combine all metrics
            combined_metrics = {
                "pipeline_type": "llm_rank_rewrite",
                "rewriter_model": self.config.model,
                "llm_metrics": {
                    "ranker": ranker_metrics,
                    "rewriter": rewriter_metrics,
                }
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
