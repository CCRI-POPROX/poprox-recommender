# src/poprox_recommender/components/scorers/llm_scorer.py
import logging
import os
from typing import List, Optional

import torch as th
from lenskit.pipeline import Component

# Import OpenAI client - you'll need to make sure openai is in your dependencies
from openai import OpenAI
from pydantic import BaseModel

from poprox_concepts import Article, CandidateSet, InterestProfile
from poprox_recommender.pytorch.decorators import torch_inference

logger = logging.getLogger(__name__)


class ScoreResponse(BaseModel):
    """Response from the LLM scoring API."""

    score: float
    explanation: str


class LLMScorerConfig(BaseModel):
    """Configuration for the LLM-based article scorer."""

    model: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 256
    api_key: Optional[str] = None
    prompt_path: Optional[str] = None


class LLMScorer(Component):
    config: LLMScorerConfig

    def __init__(self, config: LLMScorerConfig | None = None, **kwargs):
        super().__init__(config, **kwargs)

        # Initialize OpenAI client
        api_key = self.config.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key must be provided in config or as environment variable")
        self.client = OpenAI(api_key=api_key)

        # Load prompt
        self.prompt = self._load_prompt()

    def _load_prompt(self) -> str:
        """Load the ranking prompt from file or use default."""
        if self.config.prompt_path and os.path.exists(self.config.prompt_path):
            with open(self.config.prompt_path, "r") as f:
                return f.read()
        else:
            # Default prompt for article ranking
            return """
You are an advanced recommendation system that ranks news articles based on a user's reading history and interests.

Your task is to assess how well each article matches the user's preferences and assign a relevance score (0-1).

When ranking articles, consider:
1. Topic alignment with user interests
2. Similarity to previously read articles
3. Tone and framing preferences

Return only a score between 0 and 1, where 1 indicates perfect relevance to the user.
            """

    def _format_user_history(self, clicked_articles: List[Article], interest_profile: InterestProfile) -> str:
        """Format user history and interests for LLM prompt"""

        # Format articles the user has read
        history = []
        for article in clicked_articles.articles:
            article_text = f"HEADLINE: {article.headline or 'Unknown'}"
            if article.subhead:
                article_text += f"\nSUBHEAD: {article.subhead}"
            history.append(article_text)

        # Format user's explicitly stated interests
        interests = []
        for topic in interest_profile.onboarding_topics:
            if topic.preference and topic.preference > 1:  # Only include if preference > 1
                interests.append(f"{topic.entity_name} (preference: {topic.preference})")

        # Format any additional topic counts from clicks
        topic_counts = []
        if interest_profile.click_topic_counts:
            for topic, count in interest_profile.click_topic_counts.items():
                topic_counts.append(f"{topic} (clicked {count} times)")

        # Combine all information
        result = "USER READING HISTORY:\n"
        if history:
            result += "\n".join(history) + "\n\n"
        else:
            result += "No reading history available.\n\n"

        result += "USER INTERESTS:\n"
        if interests:
            result += "\n".join(interests) + "\n\n"
        else:
            result += "No explicit interests available.\n\n"

        if topic_counts:
            result += "TOPICS FROM CLICKS:\n"
            result += "\n".join(topic_counts)

        return result

    def _score_article_with_llm(self, article: Article, user_context: str) -> float:
        """Score a single article using the LLM"""

        article_text = f"HEADLINE: {article.headline or 'Unknown'}"
        if article.subhead:
            article_text += f"\nSUBHEAD: {article.subhead}"

        prompt = f"""USER CONTEXT:
{user_context}

CANDIDATE ARTICLE:
{article_text}

Please rate how relevant this article would be to this user on a scale from 0 to 1, where 1 is most relevant.
Return only the numeric score.
"""

        try:
            response = self.client.responses.parse(
                model=self.config.model,
                instructions=self.prompt,
                input=prompt,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                text_format=ScoreResponse,
            )

            # Extract the score from the response
            score = response.output_parsed.score
            return score

        except Exception as e:
            logger.error(f"Error calling LLM API: {e}")
            return 0.5  # Default score on error

    @torch_inference
    def __call__(
        self, candidate_articles: CandidateSet, clicked_articles: CandidateSet, interest_profile: InterestProfile
    ) -> CandidateSet:
        """Score candidate articles based on user history and interests."""

        # Format user context once for all articles
        user_context = self._format_user_history(clicked_articles, interest_profile)

        # Score each candidate article
        scores = []
        for article in candidate_articles.articles:
            score = self._score_article_with_llm(article, user_context)
            scores.append(score)
            logger.debug(f"Scored article '{article.headline}': {score}")

        # Convert scores to tensor and attach to candidates
        candidate_articles.scores = th.tensor(scores)

        return candidate_articles
