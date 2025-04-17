import os

import openai
from lenskit.pipeline import Component
from pydantic import BaseModel, Field

from poprox_concepts.domain import RecommendationList


class LLMRewriterConfig(BaseModel):
    """
    Configuration for the LLM-powered rewriter.
    """

    model: str = "gpt-4.1-mini"
    openai_api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))


class RewrittenOutputs(BaseModel):
    """
    Structured output: list of rewritten article texts.
    """

    texts: list[str]


class LLMRewriter(Component):
    config: LLMRewriterConfig

    def __call__(self, recommendations: RecommendationList, interest_profile: dict) -> RecommendationList:
        # configure OpenAI key
        openai.api_key = self.config.openai_api_key
        # collect texts to rewrite
        items = []
        for art in recommendations.articles:
            txt = getattr(art, "summary", None) or getattr(art, "text", None) or ""
            items.append(txt)
        prompt = (
            f"Given the user interest profile: {interest_profile}, rewrite each article to "
            "better match their interests. Respond with a JSON object with field 'texts' "
            "containing a list of the rewritten article texts. Do not include extra text. "
            "Here are the articles to rewrite:\n" + "\n---\n".join(items)
        )
        response = openai.ChatCompletion.create(model=self.config.model, messages=[{"role": "user", "content": prompt}])
        content = response.choices[0].message.content
        data = RewrittenOutputs.model_validate_json(content)
        # apply rewrites
        new_arts = []
        for art, new_txt in zip(recommendations.articles, data.texts):
            if hasattr(art, "copy"):
                try:
                    new_art = art.copy(update={"summary": new_txt})
                except TypeError:
                    setattr(art, "summary", new_txt)
                    new_art = art
            else:
                setattr(art, "summary", new_txt)
                new_art = art
            new_arts.append(new_art)
        return RecommendationList(articles=new_arts)
