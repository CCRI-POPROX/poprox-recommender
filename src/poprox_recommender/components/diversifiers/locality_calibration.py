import torch as th

from poprox_concepts import CandidateSet, InterestProfile
from poprox_concepts.domain import RecommendationList
from poprox_recommender.components.diversifiers.calibration import Calibrator
from poprox_recommender.topics import extract_locality, normalized_category_count


# Locality Calibration uses MMR
# to rerank recommendations according to
# locality calibration
class LocalityCalibrator(Calibrator):
    def __init__(self, theta: float = 0.1, num_slots=10):
        super().__init__(theta, num_slots)

    def __call__(self, candidate_articles: CandidateSet, interest_profile: InterestProfile) -> RecommendationList:
        normalized_locality_prefs = normalized_category_count(interest_profile.click_locality_counts)

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
        return RecommendationList(articles=[candidate_articles.articles[int(idx)] for idx in article_indices])

    def add_article_to_categories(self, rec_categories, article):
        locality_list = extract_locality(article)
        for locality in locality_list:
            rec_categories[locality] = rec_categories.get(locality, 0) + 1
