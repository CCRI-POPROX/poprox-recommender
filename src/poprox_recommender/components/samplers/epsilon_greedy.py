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
        shuffled = random.sample(candidates.articles, len(candidates.articles))

        for slot_idx in range(self.config.num_slots):
            selected = None

            # If our weighted coin comes up heads, try to select an article from the candidates
            if random.random() < self.config.epsilon:
                tentative = shuffled[candidate_idx]
                while tentative in selected_articles and candidate_idx < len(shuffled) - 1:
                    candidate_idx += 1
                    tentative = shuffled[candidate_idx]

                if tentative not in selected_articles:
                    selected = tentative

            # If the coin came up tails or we couldn't find an article that wasn't already selected,
            # then pick the next article from the ranked list that hasn't already been selected
            if selected is None:
                tentative = ranked.articles[ranked_idx]
                while tentative in selected_articles and ranked_idx < len(ranked.articles) - 1:
                    ranked_idx += 1
                    tentative = ranked.articles[ranked_idx]

                if tentative not in selected_articles:
                    selected = tentative

            if selected:
                selected_articles.append(selected)

        return RecommendationList(articles=selected_articles)
