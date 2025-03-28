from collections import defaultdict
from itertools import zip_longest

from lenskit.pipeline import Component
from pydantic import BaseModel

from poprox_concepts.domain import RecommendationList


class RRFConfig(BaseModel):
    num_slots: int
    k: int = 60


class ReciprocalRankFusion(Component):
    config: RRFConfig

    def __call__(self, recs1: RecommendationList, recs2: RecommendationList) -> RecommendationList:
        articles = recs1.articles
        article_scores = defaultdict(float)
        articles_by_id = {}

        recs1_extras = {article.article_id: extra or {} for article, extra in zip_longest(recs1.articles, recs1.extras)}
        recs2_extras = {article.article_id: extra or {} for article, extra in zip_longest(recs2.articles, recs2.extras)}
        recs_extras = {**recs1_extras, **recs2_extras}  # the extras in recs2 sharing same article_id will replace recs1

        for i, article in enumerate(articles, 1):
            score = 1 / (i + self.config.k)
            article_scores[article.article_id] = article_scores[article.article_id] + score
            articles_by_id[article.article_id] = article

        for i, article in enumerate(recs2.articles, 1):
            score = 1 / (i + self.config.k)
            article_scores[article.article_id] = article_scores[article.article_id] + score
            articles_by_id[article.article_id] = article

        sorted_article_scores = sorted(article_scores.items(), key=lambda x: x[1], reverse=True)

        sorted_article_ids = [article_id for article_id, article_score in sorted_article_scores]
        reciprocal_rank_fusioned_articles = [
            articles_by_id[article_id] for article_id in sorted_article_ids[: self.config.num_slots]
        ]
        rrf_extras = [recs_extras[article_id] for article_id in sorted_article_ids[: self.config.num_slots]]

        return RecommendationList(articles=reciprocal_rank_fusioned_articles, extras=rrf_extras)
