import numpy as np
from lenskit.pipeline import Component

from poprox_concepts import ArticleSet, InterestProfile


class TopkRanker(Component):
    def __init__(self, num_slots=10):
        self.num_slots = num_slots

    def __call__(self, candidate_articles: ArticleSet, interest_profile: InterestProfile) -> ArticleSet:
        articles = []
        if candidate_articles.scores is not None:
            article_indices = np.argsort(candidate_articles.scores)[-self.num_slots :][::-1]

            articles = [candidate_articles.articles[int(idx)] for idx in article_indices]

        return ArticleSet(articles=articles)
