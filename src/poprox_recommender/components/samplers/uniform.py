import random

from poprox_concepts import ArticleSet
from poprox_recommender.lkpipeline import Component


class UniformSampler(Component):
    def __init__(self, num_slots):
        self.num_slots = num_slots

    def __call__(self, candidate: ArticleSet, backup: ArticleSet | None = None) -> ArticleSet:
        backup_articles = list(
            filter(
                lambda article: article not in candidate.articles,
                backup.articles if backup else [],
            )
        )

        # we want to sample article IDs for performance
        articles = {a.article_id: a for a in candidate.articles}
        sampled = random.sample(articles.keys(), min(self.num_slots, len(candidate.articles)))

        if len(sampled) < self.num_slots and backup_articles:
            backups = {b.article_id: b for b in backup_articles}
            # add backups to articles, but allow articles to override
            articles = backups | articles
            num_backups = min(self.num_slots - len(sampled), len(backup_articles))
            sampled += random.sample(backups.keys(), num_backups)

        return ArticleSet(articles=[articles[aid] for aid in sampled])
