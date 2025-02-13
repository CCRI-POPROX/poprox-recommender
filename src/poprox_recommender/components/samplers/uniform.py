import logging
import random

from lenskit.pipeline import Component

from poprox_concepts import CandidateSet
from poprox_concepts.domain import RecommendationList

logger = logging.getLogger(__name__)


class UniformSampler(Component):
    def __init__(self, num_slots):
        self.num_slots = num_slots

    def __call__(self, candidates1: CandidateSet, candidates2: CandidateSet | None = None) -> RecommendationList:
        articles = {a.article_id: a for a in candidates1.articles}

        if candidates2 and candidates2.articles:
            backup_articles = [a for a in candidates2.articles if a.article_id not in articles]
        else:
            backup_articles = []

        logger.debug(
            "sampling %d from %d articles with %d backups", self.num_slots, len(articles), len(backup_articles)
        )

        sampled = random.sample(candidates1.articles, min(self.num_slots, len(candidates1.articles)))

        if len(sampled) < self.num_slots and backup_articles:
            num_backups = min(self.num_slots - len(sampled), len(backup_articles))
            sampled += random.sample(backup_articles, num_backups)

        return RecommendationList(articles=sampled)
