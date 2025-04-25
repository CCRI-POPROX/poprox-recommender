import os

import openai
from dotenv import load_dotenv
from lenskit.pipeline import Component
from pydantic import BaseModel, Field

from poprox_concepts import CandidateSet, InterestProfile
from poprox_concepts.domain import RecommendationList

load_dotenv()


class LlmResponse(BaseModel):
    headline: str


class LLMRewriterConfig(BaseModel):
    """
    Configuration for the LLM-powered rewriter.
    """

    model: str = "gpt-4.1"
    openai_api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))


class LLMRewriter(Component):
    config: LLMRewriterConfig

    def __call__(
        self, recommendations: RecommendationList, interest_profile: InterestProfile, clicked: CandidateSet
    ) -> RecommendationList:
        with open("prompts/rewrite.txt", "r") as f:
            prompt = f.read()

        client = openai.OpenAI(api_key=self.config.openai_api_key)

        clicked_articles = clicked.articles if clicked else []
        if not clicked_articles:
            recent_clicked_headlines = []
        else:
            recent_clicked_headlines = [art.headline for art in clicked_articles][-5:]

        # build concise profile for prompt
        clean_profile = {
            "topics": [
                t.entity_name
                for t in sorted(
                    interest_profile.onboarding_topics, key=lambda t: getattr(t, "preference", 0), reverse=True
                )
            ],
            "click_topic_counts": getattr(interest_profile, "click_topic_counts", None),
            "click_locality_counts": getattr(interest_profile, "click_locality_counts", None),
        }
        # Sort the click counts from most clicked to least clicked
        clean_profile["click_topic_counts"] = (
            [t for t, c in sorted(clean_profile["click_topic_counts"].items(), key=lambda x: x[1], reverse=True)]
            if clean_profile["click_topic_counts"]
            else []
        )
        clean_profile["click_locality_counts"] = (
            [t for t, c in sorted(clean_profile["click_locality_counts"].items(), key=lambda x: x[1], reverse=True)]
            if clean_profile["click_locality_counts"]
            else []
        )

        # rewrite article headlines
        for art in recommendations.articles:
            # build prompt
            input_txt = f"""User topics of interest (from most to least important): {", ".join(clean_profile["topics"])}
            Recent clicked articles: {", ".join(recent_clicked_headlines)}
            Clicked article topics: {", ".join(clean_profile["click_topic_counts"])}
            Clicked article localities: {", ".join(clean_profile["click_locality_counts"])}

            Headline to rewrite: {art.headline}
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
