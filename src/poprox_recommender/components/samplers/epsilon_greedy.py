import logging
import random

from lenskit.pipeline import Component
from pydantic import BaseModel

from poprox_concepts import CandidateSet
from poprox_concepts.domain import RecommendationList

logger = logging.getLogger(__name__)


class EpsilonGreedyConfig(BaseModel):
    num_slots: int = 10
    epsilon: float = 0.1


class EpsilonGreedy(Component):
    config: EpsilonGreedyConfig

    def __call__(self, ranked: RecommendationList, candidates: CandidateSet) -> RecommendationList:
        selected_articles = []
        ranked_idx = 0
        candidate_idx = 0

        # Shuffle the candidates so taking the first one amounts to sampling uniformly at random
        candidates.articles = random.sample(candidates.articles, len(candidates.articles))

        for slot_idx in range(self.config.num_slots):
            if random.random() < self.config.epsilon:
                selected_articles.append(candidates.articles[candidate_idx])
                candidate_idx += 1
            else:
                selected_articles.append(ranked.articles[ranked_idx])
                ranked_idx += 1

        return RecommendationList(articles=selected_articles)
