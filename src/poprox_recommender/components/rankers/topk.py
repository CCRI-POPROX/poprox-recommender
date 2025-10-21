import numpy as np
from lenskit.pipeline import Component
from pydantic import BaseModel

from poprox_concepts.domain import CandidateSet, ImpressedRecommendations


class TopkConfig(BaseModel):
    num_slots: int = 10


class TopkRanker(Component):
    config: TopkConfig

    def __call__(self, candidate_articles: CandidateSet) -> ImpressedRecommendations:
        articles = []
        if candidate_articles.scores is not None:
            article_indices = np.argsort(candidate_articles.scores)[-self.config.num_slots :][::-1]

            articles = [candidate_articles.articles[int(idx)] for idx in article_indices]

        return ImpressedRecommendations.from_articles(articles)
