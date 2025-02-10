import numpy as np
from lenskit.pipeline import Component

from poprox_concepts import CandidateSet, InterestProfile
from poprox_concepts.domain import RecommendationList


class TopkRanker(Component):
    def __init__(self, num_slots=10):
        self.num_slots = num_slots

    def __call__(self, candidate_articles: CandidateSet, interest_profile: InterestProfile) -> RecommendationList:
        articles = []
        if candidate_articles.scores is not None:
            article_indices = np.argsort(candidate_articles.scores)[-self.num_slots :][::-1]

            articles = [candidate_articles.articles[int(idx)] for idx in article_indices]

        return RecommendationList(articles=articles)
