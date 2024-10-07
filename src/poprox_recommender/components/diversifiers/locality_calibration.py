import logging
from collections import defaultdict

import torch as th

from poprox_concepts import Article, ArticleSet, InterestProfile
from poprox_recommender.components.diversifiers.calibration import Calibrator
from poprox_recommender.components.samplers import SoftmaxSampler
from poprox_recommender.topics import extract_locality, normalized_category_count

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# Locality Calibration uses MMR
# to rerank recommendations according to
# locality calibration
class LocalityCalibrator(Calibrator):
    def __init__(self, theta: float = 0.1, num_slots=10):  # , scorer=None):
        super().__init__(theta, num_slots)
        # add a different scorer than NMRS. Theta
        self.scorer = SoftmaxSampler(num_slots=num_slots, temperature=30.0)

    def __call__(self, candidate_articles: ArticleSet, interest_profile: InterestProfile) -> ArticleSet:
        if self.scorer is not None:
            candidate_articles = self.scorer.sort_score(candidate_articles)
        normalized_locality_prefs = normalized_category_count(
            self.article_locality_distribution(candidate_articles.articles)
        )
        logger.info(f"nomarlized locality distribution of canidate articles: {normalized_locality_prefs}")

        if candidate_articles.scores is not None:
            article_scores = th.sigmoid(th.tensor(candidate_articles.scores))
        else:
            article_scores = th.zeros(len(candidate_articles.articles))

        article_scores = article_scores.cpu().detach().numpy()

        article_indices = self.calibration(
            article_scores,
            candidate_articles.articles,
            normalized_locality_prefs,
            self.theta,
            topk=self.num_slots,
        )
        # new_calibrated_indicies = [idx for idx in article_indices if idx >= self.num_slots]
        logger.info(f"Articles added as a results of calibration: {article_indices}")

        return ArticleSet(articles=[candidate_articles.articles[int(idx)] for idx in article_indices])

    def add_article_to_categories(self, rec_categories, article):
        locality_list = extract_locality(article)
        for locality in locality_list:
            rec_categories[locality] = rec_categories.get(locality, 0) + 1

    def article_locality_distribution(self, articles: list[Article]):
        locality_count_dict = defaultdict(int)

        for article in articles:
            article_locality = extract_locality(article)
            for locality in article_locality:
                locality_count_dict[locality] += 1

        return locality_count_dict
