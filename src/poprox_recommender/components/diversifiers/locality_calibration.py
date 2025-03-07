import logging
from collections import defaultdict

import torch as th
from pydantic import BaseModel

from poprox_concepts.domain import Article, CandidateSet, InterestProfile
from poprox_recommender.components.diversifiers.calibration import Calibrator, compute_kl_divergence
from poprox_recommender.topics import (
    extract_general_topics,
    extract_locality,
    normalized_category_count,
)

LOCALITY_DISTANCE_THRESHOLD = 0.1


logger = logging.getLogger(__name__)


class LocalityCalibratorConfig(BaseModel):
    theta_locality: float = 0.55
    theta_topic: float = 0.3
    num_slots: int = 10


class LocalityCalibrator(Calibrator):
    config: LocalityCalibratorConfig

    def __call__(
        self,
        candidate_articles: CandidateSet,
        interest_profile: InterestProfile,
        theta_topic: float | None,
        theta_locality: float | None,
    ) -> CandidateSet:
        theta_topic = self.config.theta_topic if theta_topic is None else theta_topic
        theta_locality = self.config.theta_locality if theta_locality is None else theta_locality

        normalized_topic_prefs = LocalityCalibrator.compute_topic_prefs(interest_profile)
        normalized_locality_prefs = self.compute_local_prefs(candidate_articles)

        if candidate_articles.scores is not None:
            article_scores = th.sigmoid(th.tensor(candidate_articles.scores))
        else:
            article_scores = th.zeros(len(candidate_articles.articles))

        article_scores = article_scores.cpu().detach().numpy()

        (
            article_indices,
            topic_only_article_indices,
            final_calibrations,
            localities_outside_threshold,
        ) = self.calibration(
            article_scores,
            candidate_articles.articles,
            normalized_topic_prefs,
            normalized_locality_prefs,
            theta_topic,
            theta_locality,
            topk=self.config.num_slots,
        )

        selected = CandidateSet(
            articles=[candidate_articles.articles[idx] for idx in article_indices]
        )  # all selected articles

        selected.treatment_flags = [index not in topic_only_article_indices for index in article_indices]
        selected.k1_topic = final_calibrations[0]
        selected.k1_locality = final_calibrations[1]

        selected.is_inside_locality_threshold = not localities_outside_threshold
        return selected

    def add_article_to_categories(self, rec_topics, article):
        rec_topics = rec_topics.copy()
        topics = extract_general_topics(article)
        for topic in topics:
            rec_topics[topic] = rec_topics[topic] + 1
        return rec_topics

    def add_article_to_localities(self, rec_localities, article):
        rec_localities = rec_localities.copy()
        localities = extract_locality(article)
        for local in localities:
            rec_localities[local] = rec_localities[local] + 1
        return rec_localities

    def normalized_categories_with_candidate(self, rec_categories, article):
        rec_categories_with_candidate = rec_categories.copy()
        rec_categories_with_candidate = self.add_article_to_categories(rec_categories_with_candidate, article)
        return normalized_category_count(rec_categories_with_candidate)

    def normalized_localities_with_candidate(self, rec_localities, article):
        rec_localities_with_candidate = rec_localities.copy()
        rec_localities_with_candidate = self.add_article_to_localities(rec_localities_with_candidate, article)
        return normalized_category_count(rec_localities_with_candidate)

    def calibration(
        self,
        relevance_scores,
        articles,
        topic_preferences,
        locality_preferences,
        theta_topic,
        theta_locality,
        topk,
    ) -> list[Article]:
        # MR_i = (1 - theta_topic - theta_local) * reward_i - theta_topic * C_topic - theta_local * C_local
        # R is all candidates (not selected yet)
        recommendations = []  # final recommendation (topk index)
        topic_only_recommendations = []
        topic_categories = defaultdict(int)
        topic_only_categories = defaultdict(int)
        local_categories = defaultdict(int)

        final_calibrations = [None, None]

        for _ in range(topk):
            candidate = None  # next item
            topic_candidate = None
            best_candidate_score = float("-inf")
            best_topic_candidate_score = float("-inf")

            for article_idx, article_score in enumerate(relevance_scores):  # iterate R for next item
                if article_idx in recommendations:
                    continue

                normalized_candidate_topics = self.normalized_categories_with_candidate(
                    topic_categories, articles[article_idx]
                )
                normalized_topic_candidate_topics = self.normalized_categories_with_candidate(
                    topic_only_categories, articles[article_idx]
                )
                normalized_candidate_locality = self.normalized_localities_with_candidate(
                    local_categories, articles[article_idx]
                )

                calibration_topic = compute_kl_divergence(topic_preferences, normalized_candidate_topics)
                calibration_topic_only = compute_kl_divergence(topic_preferences, normalized_topic_candidate_topics)
                calibration_local = compute_kl_divergence(locality_preferences, normalized_candidate_locality)

                # TODO or other MOE
                adjusted_candidate_score = (
                    (1 - theta_locality - theta_topic) * article_score
                    - (theta_topic * calibration_topic)
                    - (theta_locality * calibration_local)
                )
                adjusted_topic_candidate_score = (1 - theta_locality - theta_topic) * article_score - (
                    (theta_topic) * calibration_topic_only
                )
                if adjusted_candidate_score > best_candidate_score:
                    best_candidate_score = adjusted_candidate_score
                    candidate = article_idx
                    final_calibrations = [calibration_topic, calibration_local]

                if adjusted_topic_candidate_score > best_topic_candidate_score:
                    best_topic_candidate_score = adjusted_topic_candidate_score
                    topic_candidate = article_idx

            if candidate is not None:
                recommendations.append(candidate)
                topic_categories = self.add_article_to_categories(topic_categories, articles[candidate])
                local_categories = self.add_article_to_localities(local_categories, articles[candidate])

            if topic_candidate is not None:
                topic_only_recommendations.append(topic_candidate)
                topic_only_categories = self.add_article_to_categories(topic_only_categories, articles[topic_candidate])

        # logger.info(f"Rec'ed newsletter distribution {local_categories}")
        # logger.info(f"Todays news  distribution {locality_preferences}")
        normalized_local_categories = normalized_category_count(local_categories)
        localities_outside_threshold = [
            locality
            for locality in normalized_local_categories
            if locality in locality_preferences
            and abs(normalized_local_categories[locality] - locality_preferences[locality])
            > LOCALITY_DISTANCE_THRESHOLD
        ]

        return (
            recommendations,
            topic_only_recommendations,
            final_calibrations,
            localities_outside_threshold,
        )

    def compute_local_prefs(self, candidate_articles: CandidateSet):
        locality_preferences: dict[str, int] = defaultdict(int)
        articles = candidate_articles.articles

        for article in articles:
            candidate_locality = extract_locality(article) or set()
            for locality in candidate_locality:
                locality_preferences[locality] += 1

        normalized_locality_pres = normalized_category_count(locality_preferences)
        return normalized_locality_pres

    @staticmethod
    def compute_topic_prefs(interest_profile):
        topic_preferences: dict[str, int] = defaultdict(int)
        # TODO uncomment to verify interest profile bug
        # logger.info(f"Interest Profile {interest_profile.click_topic_counts}")
        for interest in interest_profile.onboarding_topics:
            topic_preferences[interest.entity_name] = max(interest.preference - 1, 0)

        if interest_profile.click_topic_counts:
            for topic, click_count in interest_profile.click_topic_counts.items():
                topic_preferences[topic] = topic_preferences.get(topic, 0) + click_count
        normalized_topic_prefs = normalized_category_count(topic_preferences)
        return normalized_topic_prefs
