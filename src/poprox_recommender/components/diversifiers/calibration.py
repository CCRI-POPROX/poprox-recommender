from collections import defaultdict

import numpy as np
from lenskit.pipeline import Component
from pydantic import BaseModel

from poprox_concepts.domain import Article
from poprox_recommender.topics import normalized_category_count


class CalibratorConfig(BaseModel):
    theta: float = 0.1
    num_slots: int = 10


# General calibration uses MMR
# to rerank recommendations according to
# certain calibration context (e.g. news topic, locality)
class Calibrator(Component):
    config: CalibratorConfig

    def add_article_to_categories(self, rec_categories_with_candidate, article):
        pass

    def normalized_categories_with_candidate(self, rec_categories, article):
        rec_categories_with_candidate = rec_categories.copy()
        self.add_article_to_categories(rec_categories_with_candidate, article)
        return normalized_category_count(rec_categories_with_candidate)

    def calibration(self, relevance_scores, articles, preferences, theta, topk) -> list[Article]:
        # MR_i = \theta * reward_i - (1 - \theta)*C(S + i) # C is calibration
        # R is all candidates (not selected yet)

        recommendations = []  # final recommendation (topk index)
        rec_categories = defaultdict(int)  # frequency distribution of categories of S

        for _ in range(topk):
            candidate = None  # next item
            best_candidate_score = float("-inf")

            for article_idx, article_score in enumerate(relevance_scores):  # iterate R for next item
                if article_idx in recommendations:
                    continue

                normalized_candidate_topics = self.normalized_categories_with_candidate(
                    rec_categories, articles[article_idx]
                )
                calibration = compute_kl_divergence(preferences, normalized_candidate_topics)

                adjusted_candidate_score = (1 - theta) * article_score - (theta * calibration)
                if adjusted_candidate_score > best_candidate_score:
                    best_candidate_score = adjusted_candidate_score
                    candidate = article_idx

            if candidate is not None:
                recommendations.append(candidate)
                self.add_article_to_categories(rec_categories, articles[candidate])

        return recommendations


# from https://github.com/CCRI-POPROX/poprox-recommender/blob/feature/experiment0/tests/test_calibration.ipynb
def compute_kl_divergence(interacted_distr, reco_distr, kl_div=0.0, alpha=0.01):
    """
    KL (p || q), the lower the better.

    alpha is not really a tuning parameter, it's just there to make the
    computation more numerically stable.
    """
    for category, score in interacted_distr.items():
        reco_score = reco_distr.get(category, 0.0)
        reco_score = (1 - alpha) * reco_score + alpha * score
        if reco_score != 0.0:
            kl_div += score * np.log2(score / reco_score)

    return kl_div
