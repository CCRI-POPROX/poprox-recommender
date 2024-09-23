from collections import defaultdict

import torch as th

from poprox_concepts import Article, ArticleSet, InterestProfile
from poprox_recommender.diversifiers import Calibrator, compute_kl_divergence
from poprox_recommender.lkpipeline import Component
from poprox_recommender.topics import extract_locality, normalized_topic_count


# Locality Calibration uses MMR
# to rerank recommendations according to
# locality calibration
class LocalityCalibrator(Component, Calibrator):
    def __init__(self, theta: float = 0.1, num_slots=10):
        super.__init__(theta, num_slots)

    def __call__(self, candidate_articles: ArticleSet, interest_profile: InterestProfile) -> ArticleSet:
        normalized_locality_prefs = normalized_topic_count(interest_profile.click_locality_counts)

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
        return ArticleSet(articles=[candidate_articles.articles[int(idx)] for idx in article_indices])

    def calibration(self, relevance_scores, articles, preferences, theta, topk) -> list[Article]:
        # MR_i = \theta * reward_i - (1 - \theta)*C(S + i) # C is calibration
        # R is all candidates (not selected yet)

        recommendations = []  # final recommendation (topk index)
        rec_categories = defaultdict(int)  # frequency distribution of catregories of S

        for k in range(topk):
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

    def normalized_categories_with_candidate(self, rec_categories, article):
        rec_categories_with_candidate = rec_categories.copy()
        self.add_article_to_categories(rec_categories_with_candidate, article)
        return normalized_topic_count(rec_categories_with_candidate)

    def add_article_to_categories(self, rec_categories, article):
        locality_list = extract_locality(article)
        for locality in locality_list:
            rec_categories[locality] = rec_categories.get(locality, 0) + 1
