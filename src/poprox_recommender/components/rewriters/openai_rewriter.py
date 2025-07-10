import os

import openai
from dotenv import load_dotenv
from lenskit.pipeline import Component
from pydantic import BaseModel, Field

from poprox_concepts.domain import RecommendationList

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

    def __call__(self, recommendations: RecommendationList, user_model: str) -> RecommendationList:
        with open("prompts/rewrite.md", "r") as f:
            prompt = f.read()

        client = openai.OpenAI(api_key=self.config.openai_api_key)

        # rewrite article headlines
        for art in recommendations.articles:
            # build prompt using the user_model from LLMRanker
            input_txt = f"""User interest profile:
{user_model}

Headline to rewrite:
{art.headline}

Article text:
{art.body}
"""

            response = client.responses.parse(
                model=self.config.model,
                instructions=prompt,
                input=input_txt,
                temperature=0.5,
                text_format=LlmResponse,
            )
            # Update the article headline with the rewritten one
            art.headline = response.output_parsed.headline

        return recommendations
