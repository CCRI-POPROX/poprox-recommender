import os

import openai
from lenskit.pipeline import Component
from pydantic import BaseModel, Field

from poprox_concepts import CandidateSet
from poprox_concepts.domain import RecommendationList


class LLMRankerConfig(BaseModel):
    """
    Configuration for the LLM-powered ranker.
    """

    model: str = "gpt-4.1-mini"
    num_slots: int = 10
    openai_api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))


class RankedIndices(BaseModel):
    """
    Structured output: list of selected indices from the candidate pool.
    """

    indices: list[int]


class LLMRanker(Component):
    config: LLMRankerConfig

    def __call__(self, candidate_articles: CandidateSet, interest_profile: dict) -> RecommendationList:
        # configure OpenAI key
        openai.api_key = self.config.openai_api_key
        # summarize candidates for the prompt
        items = []
        for i, art in enumerate(candidate_articles.articles):
            content = getattr(art, "summary", None) or getattr(art, "text", None) or ""
            items.append(f"{i}: {art.title} - {content}")
        prompt = (
            f"Given the user interest profile: {interest_profile}, select the top "
            f"{self.config.num_slots} articles by relevance. "
            "Respond with a JSON object with field 'indices' containing a list of the chosen article indices. "
            "Here are the candidate articles:\n" + "\n".join(items)
        )
        response = openai.ChatCompletion.create(model=self.config.model, messages=[{"role": "user", "content": prompt}])
        content = response.choices[0].message.content
        data = RankedIndices.model_validate_json(content)
        selected = [candidate_articles.articles[i] for i in data.indices]
        return RecommendationList(articles=selected)
