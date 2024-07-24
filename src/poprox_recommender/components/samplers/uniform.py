import random

from poprox_concepts import ArticleSet


class UniformSampler:
    def __init__(self, num_slots):
        self.num_slots = num_slots

    def __call__(self, candidate: ArticleSet, backup: ArticleSet | None = None) -> ArticleSet:
        backup_articles = list(filter(lambda article: article not in candidate.articles, backup.articles))

        num_backups = (
            self.num_slots - len(candidate.articles)
            if len(candidate.articles) + len(backup_articles) >= self.num_slots
            else len(backup_articles)
        )

        if len(candidate.articles) < self.num_slots and backup_articles:
            sampled = random.sample(candidate.articles, len(candidate.articles)) + random.sample(
                backup_articles, num_backups
            )
        else:
            sampled = random.sample(candidate.articles, self.num_slots)

        return ArticleSet(articles=sampled)
