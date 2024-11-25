from collections import defaultdict

import torch as th

from poprox_concepts import Article, ArticleSet, InterestProfile
from poprox_recommender.components.diversifiers.calibration import compute_kl_divergence
from poprox_recommender.lkpipeline import Component
from poprox_recommender.paths import model_file_path
from poprox_recommender.topics import extract_general_topics, extract_locality, normalized_category_count

# Only uncomment this in offline theta value exploration
# KL_VALUE_PATH = '/home/sun00587/research/News_Locality_Polarization/poprox-recommender-locality/outputs/theta_kl_values_11-17.txt'


class LocalityCalibrator(Component):
    def __init__(self, theta_local: float = 0.1, theta_topic: float = 0.1, num_slots=10):
        """
        TODOs: If set different theta_topic and theta_local values for different users,
        then can save them in interest_profile
        """
        self.theta_local = theta_local
        self.theta_topic = theta_topic
        self.num_slots = num_slots

    def __call__(
        self,
        candidate_articles: ArticleSet,
        interest_profile: InterestProfile,
    ) -> ArticleSet:
        normalized_topic_prefs = self.compute_topic_prefs(interest_profile)
        normalized_locality_prefs = self.compute_local_prefs(candidate_articles)

        if candidate_articles.scores is not None:
            article_scores = th.sigmoid(th.tensor(candidate_articles.scores))
        else:
            article_scores = th.zeros(len(candidate_articles.articles))

        article_scores = article_scores.cpu().detach().numpy()

        article_indices, final_calibrations = self.calibration(
            article_scores,
            candidate_articles.articles,
            normalized_topic_prefs,
            normalized_locality_prefs,
            self.theta_topic,
            self.theta_local,
            topk=self.num_slots,
        )

        # Save computed kl divergence for topic and locality
        # Only uncomment this in offline theta value exploration
        # with open(KL_VALUE_PATH, 'a') as file:
        #     file.write('{}_top_{}_loc_{},{},{}\n'.format(str(interest_profile.profile_id), theta_topic, theta_locality, final_calibrations[0], final_calibrations[1]))

        return ArticleSet(
            articles=[candidate_articles.articles[idx] for idx in article_indices]
        )  # all selected articles

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
        self, relevance_scores, articles, topic_preferences, locality_preferences, theta_topic, theta_local, topk
    ) -> list[Article]:
        # MR_i = (1 - theta_topic - theta_local) * reward_i - theta_topic * C_topic - theta_local * C_local
        # R is all candidates (not selected yet)

        recommendations = []  # final recommendation (topk index)

        topic_categories = defaultdict(int)
        local_categories = defaultdict(int)

        final_calibrations = [None, None]

        for _ in range(topk):
            candidate = None  # next item
            best_candidate_score = float("-inf")

            for article_idx, article_score in enumerate(relevance_scores):  # iterate R for next item
                if article_idx in recommendations:
                    continue

                normalized_candidate_topics = self.normalized_categories_with_candidate(
                    topic_categories, articles[article_idx]
                )
                normalized_candidate_locality = self.normalized_localities_with_candidate(
                    local_categories, articles[article_idx]
                )

                calibration_topic = compute_kl_divergence(topic_preferences, normalized_candidate_topics)
                calibration_local = compute_kl_divergence(locality_preferences, normalized_candidate_locality)

                # TODO or other MOE
                adjusted_candidate_score = (
                    (1 - theta_local - theta_topic) * article_score
                    - (theta_topic * calibration_topic)
                    - (theta_local * calibration_local)
                )
                if adjusted_candidate_score > best_candidate_score:
                    best_candidate_score = adjusted_candidate_score
                    candidate = article_idx
                    final_calibrations = [calibration_topic, calibration_local]

            if candidate is not None:
                recommendations.append(candidate)
                topic_categories = self.add_article_to_categories(topic_categories, articles[candidate])
                local_categories = self.add_article_to_localities(local_categories, articles[candidate])

        return recommendations, final_calibrations

    def compute_local_prefs(self, candidate_articles: ArticleSet):
        locality_preferences: dict[str, int] = defaultdict(int)
        candidate_articles = candidate_articles.articles

        for article in candidate_articles:
            candidate_locality = extract_locality(article) or set()
            for locality in candidate_locality:
                locality_preferences[locality] += 1

        normalized_locality_pres = normalized_category_count(locality_preferences)
        return normalized_locality_pres

    def compute_topic_prefs(self, interest_profile):
        topic_preferences: dict[str, int] = defaultdict(int)

        for interest in interest_profile.onboarding_topics:
            topic_preferences[interest.entity_name] = max(interest.preference - 1, 0)

        if interest_profile.click_topic_counts:
            for topic, click_count in interest_profile.click_topic_counts.items():
                topic_preferences[topic] += click_count

        normalized_topic_prefs = normalized_category_count(topic_preferences)
        return normalized_topic_prefs
