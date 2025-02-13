from collections import defaultdict

from lenskit.pipeline import Component

from poprox_concepts.domain import RecommendationList


class ReciprocalRankFusion(Component):
    def __init__(self, num_slots: int, k: int = 60):
        self.num_slots = num_slots
        self.k = k

    def __call__(self, recs1: RecommendationList, recs2: RecommendationList) -> RecommendationList:
        articles = recs1.articles
        article_scores = defaultdict(float)
        articles_by_id = {}

        for i, article in enumerate(articles, 1):
            score = 1 / (i + self.k)
            article_scores[article.article_id] = article_scores[article.article_id] + score
            articles_by_id[article.article_id] = article

        for i, article in enumerate(recs2.articles, 1):
            score = 1 / (i + self.k)
            article_scores[article.article_id] = article_scores[article.article_id] + score
            articles_by_id[article.article_id] = article

        sorted_article_scores = sorted(article_scores.items(), key=lambda x: x[1], reverse=True)

        sorted_article_ids = [article_id for article_id, article_score in sorted_article_scores]
        reciprocal_rank_fusioned_articles = [
            articles_by_id[article_id] for article_id in sorted_article_ids[: self.num_slots]
        ]

        return RecommendationList(articles=reciprocal_rank_fusioned_articles)
