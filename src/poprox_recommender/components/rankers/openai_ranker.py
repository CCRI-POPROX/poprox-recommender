import os

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

    def _structure_interest_profile(self, interest_profile: InterestProfile, articles_clicked: CandidateSet) -> str:
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
        response = client.responses.create(
            model="gpt-4.1-2025-04-14",
            instructions=prompt,
            input=interest_profile_str,
        )

        return response.output_text

    def __call__(
        self, candidate_articles: CandidateSet, interest_profile: InterestProfile, articles_clicked: CandidateSet
    ) -> RecommendationList:
        with open("prompts/rank.txt", "r") as f:
            prompt = f.read()

        client = openai.OpenAI(api_key=self.config.openai_api_key)

        # build concise profile for prompt
        profile_str = self._structure_interest_profile(interest_profile, articles_clicked)
        # build user model
        user_model = self._build_user_model(profile_str)

        # summarize candidates for the prompt
        items = []
        for i, art in enumerate(candidate_articles.articles):
            items.append(f"{i}: {art.headline} - {art.subhead}")

        input_txt = f"""User model:
{user_model}

{profile_str}

Candidate articles:
{", ".join(items)}

Make sure you select EXACTLY {self.config.num_slots} articles from the candidate pool.
"""

        response = client.responses.parse(
            model=self.config.model, instructions=prompt, input=input_txt, text_format=LlmResponse, temperature=0.5
        )

        response = response.output_parsed
        selected = [candidate_articles.articles[i] for i in response.recommended_article_ids]
        return RecommendationList(articles=selected)


# TODOs
# - Updated prompt to reflect input data
# - Pass through to rewriter
# - Updated rewriter prompt to reflect input data
