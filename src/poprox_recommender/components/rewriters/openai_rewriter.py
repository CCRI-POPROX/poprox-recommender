import json
import os
from typing import List, Literal, Optional

import openai
from lenskit.pipeline import Component
from pydantic import BaseModel, Field

from poprox_concepts.domain import Article, RecommendationList


class LLMRewriterConfig(BaseModel):
    """
    Configuration for the LLM-powered rewriter.
    """

    model: str = "gpt-4.1-mini"
    openai_api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    fields_to_rewrite: List[Literal["headline", "subhead", "body", "summary"]] = ["summary"]


class RewrittenArticle(BaseModel):
    """
    Structured output for a single rewritten article.
    """

    headline: Optional[str] = None
    subhead: Optional[str] = None
    body: Optional[str] = None
    summary: Optional[str] = None


class RewrittenOutputs(BaseModel):
    """
    Structured output: list of rewritten article fields.
    """

    articles: List[RewrittenArticle]


class LLMRewriter(Component):
    config: LLMRewriterConfig

    def __call__(self, recommendations: RecommendationList, interest_profile: dict) -> RecommendationList:
        # configure OpenAI key
        openai.api_key = self.config.openai_api_key

        # collect article data to rewrite
        items = []
        for art in recommendations.articles:
            article_data = {}
            for field in self.config.fields_to_rewrite:
                article_data[field] = getattr(art, field, None) or ""
            items.append(article_data)

        prompt = (
            f"Given the user interest profile: {interest_profile}, rewrite the specified fields "
            f"of each article to better match interests. The fields to rewrite are: {self.config.fields_to_rewrite}. "
            "Respond with a JSON object with field 'articles' containing a list of objects, each with "
            f"only the rewritten fields ({', '.join(self.config.fields_to_rewrite)}). "
            "Do not include extra text. Here are the article fields to rewrite:\n" + json.dumps(items, indent=2)
        )

        response = openai.ChatCompletion.create(model=self.config.model, messages=[{"role": "user", "content": prompt}])
        content = response.choices[0].message.content
        data = RewrittenOutputs.model_validate_json(content)

        # apply rewrites while ensuring immutability
        new_arts = []
        for art, rewritten in zip(recommendations.articles, data.articles):
            # Create a dictionary from the original article
            if hasattr(art, "model_dump"):
                article_data = art.model_dump()
            else:
                # Fallback for non-pydantic models
                article_data = {
                    k: getattr(art, k) for k in dir(art) if not k.startswith("_") and not callable(getattr(art, k))
                }

            # Update with rewritten fields
            for field in self.config.fields_to_rewrite:
                rewritten_value = getattr(rewritten, field)
                if rewritten_value:
                    article_data[field] = rewritten_value

            # Create a new Article instance
            if hasattr(Article, "parse_obj"):
                # Legacy Pydantic v1
                new_art = Article.parse_obj(article_data)
            elif hasattr(Article, "model_validate"):
                # Pydantic v2
                new_art = Article.model_validate(article_data)
            else:
                # Generic fallback (less ideal)
                new_art = type(art)(**article_data)

            new_arts.append(new_art)

        return RecommendationList(articles=new_arts)
