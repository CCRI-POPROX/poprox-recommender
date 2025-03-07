import numpy as np
from lenskit.pipeline import Component
from pydantic import BaseModel

from poprox_concepts import CandidateSet, InterestProfile
from poprox_concepts.domain import RecommendationList


class TopkConfig(BaseModel):
    num_slots: int = 10


class TopkRanker(Component):
    config: TopkConfig

    def __call__(self, candidate_articles: CandidateSet, interest_profile: InterestProfile) -> RecommendationList:
        articles = []
        if candidate_articles.scores is not None:
            article_indices = np.argsort(candidate_articles.scores)[-self.config.num_slots :][::-1]

            articles = [candidate_articles.articles[int(idx)] for idx in article_indices]

        return RecommendationList(articles=articles)
