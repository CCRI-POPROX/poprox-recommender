from collections import defaultdict

from poprox_concepts import ArticleSet
from poprox_recommender.lkpipeline import Component
from poprox_recommender.lkpipeline.types import Lazy


class ReciprocalRankFusion(Component):
    def __init__(self, num_slots: int):
        self.num_slots = num_slots

    def __call__(self, candidates1: ArticleSet, candidates2: Lazy[ArticleSet]) -> ArticleSet:
        articles = candidates1.articles
        article_scores = defaultdict(float)
        articles_by_id = {}

        i = 1
        k = 60
        for article in articles:
            score = 1 / (i + k)
            article_scores[article.article_id] = article_scores[article.article_id] + score
            articles_by_id[article.article_id] = article
            i += 1

        i = 1
        for article in candidates2.get().articles:
            score = 1 / (i + k)
            article_scores[article.article_id] = article_scores[article.article_id] + score
            articles_by_id[article.article_id] = article
            i += 1

        sorted_article_scores = sorted(article_scores.items(), key=lambda x: x[1], reverse=True)

        sorted_article_ids = [article_id for article_id, article_score in sorted_article_scores]

        reciprocal_rank_fusioned_articles = []
        for article_id in sorted_article_ids:
            reciprocal_rank_fusioned_articles.append(articles_by_id[article_id])

        # Return the resulting ArticleSet, limiting the size to num_slots
        return ArticleSet(articles=reciprocal_rank_fusioned_articles[: self.num_slots])
