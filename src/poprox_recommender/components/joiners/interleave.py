from itertools import zip_longest

from lenskit.pipeline import Component

from poprox_concepts.domain import RecommendationList


class Interleave(Component):
    config: None

    def __call__(self, recs1: RecommendationList, recs2: RecommendationList) -> RecommendationList:
        articles = []
        extras = []

        recs1_extras = {article.article_id: extra for article, extra in zip_longest(recs1.articles, recs1.extras)}
        recs2_extras = {article.article_id: extra for article, extra in zip_longest(recs2.articles, recs2.extras)}
        recs_extras = {**recs1_extras, **recs2_extras}  # the extras in recs2 sharing same article_id will replace recs1

        for pair in zip_longest(recs1.articles, recs2.articles):
            for article in pair:
                if article is not None:
                    articles.append(article)
                    extra = recs_extras[article.article_id] or {}
                    extras.append(extra)

        return RecommendationList(articles=articles, extras=extras)
