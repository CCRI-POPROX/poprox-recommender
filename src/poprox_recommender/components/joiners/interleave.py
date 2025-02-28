from itertools import zip_longest

from lenskit.pipeline import Component

from poprox_concepts.domain import RecommendationList


class Interleave(Component):
    config: None

    def __call__(self, recs1: RecommendationList, recs2: RecommendationList) -> RecommendationList:
        articles = []
        for pair in zip_longest(recs1.articles, recs2.articles):
            for article in pair:
                if article is not None:
                    articles.append(article)

        return RecommendationList(articles=articles)
