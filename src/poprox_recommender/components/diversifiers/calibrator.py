import numpy as np

from poprox_recommender.lkpipeline import Component


# General calibration uses MMR
# to rerank recommendations according to
# certain calibration context (e.g. news topic, locality)
class Calibrator(Component):
    def __init__(self, theta: float = 0.1, num_slots=10):
        # Theta term controls the score and calibration tradeoff, the higher
        # the theta the higher the resulting recommendation will be calibrated.
        self.theta = theta
        self.num_slots = num_slots

    def __call__():
        pass

    def calibration():
        pass

    def add_article_to_categories():
        pass

    def normalized_categories_with_candidate():
        pass


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
