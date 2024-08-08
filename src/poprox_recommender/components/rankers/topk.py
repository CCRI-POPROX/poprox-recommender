from typing import Any

import numpy as np

from poprox_concepts import ArticleSet, InterestProfile
from poprox_recommender.components.rankers import Ranker


class TopkRanker(Ranker):
    def __init__(self, algo_params: dict[str, Any], num_slots=10):
        self.validate_algo_params(algo_params, [])
        self.num_slots = num_slots

    def __call__(self, candidate_articles: ArticleSet, interest_profile: InterestProfile) -> ArticleSet:
        articles = []
        if candidate_articles.scores is not None:
            article_indices = np.argsort(candidate_articles.scores)[-self.num_slots :][::-1]

            articles = [candidate_articles.articles[int(idx)] for idx in article_indices]

        return ArticleSet(articles=articles)
