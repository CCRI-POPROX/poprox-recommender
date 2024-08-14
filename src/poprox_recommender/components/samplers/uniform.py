import random

from poprox_concepts import ArticleSet
from poprox_recommender.lkpipeline import Component


class UniformSampler(Component):
    def __init__(self, num_slots):
        self.num_slots = num_slots

    def __call__(self, candidate: ArticleSet, backup: ArticleSet | None = None) -> ArticleSet:
        articles = {a.article_id: a for a in candidate.articles}

        if backup.articles:
            backup_articles = [a for a in backup.articles if a.article_id not in articles]
        else:
            backup_articles = []

        sampled = random.sample(candidate.articles, min(self.num_slots, len(candidate.articles)))

        if len(sampled) < self.num_slots and backup_articles:
            num_backups = min(self.num_slots - len(sampled), len(backup_articles))
            sampled += random.sample(backup_articles, num_backups)

        return ArticleSet(articles=sampled)
