from collections import defaultdict

import torch as th

from poprox_concepts import Article, ArticleSet, InterestProfile
from poprox_recommender.diversifiers import Calibrator, compute_kl_divergence
from poprox_recommender.lkpipeline import Component
from poprox_recommender.topics import extract_general_topics, normalized_topic_count


# Topic Calibration uses MMR
# to rerank recommendations according to
# topic calibration
class TopicCalibrator(Component, Calibrator):
    def __call__(self, candidate_articles: ArticleSet, interest_profile: InterestProfile) -> ArticleSet:
        normalized_topic_prefs = self.compute_topic_dist(interest_profile)

        if candidate_articles.scores is not None:
            article_scores = th.sigmoid(th.tensor(candidate_articles.scores))
        else:
            article_scores = th.zeros(len(candidate_articles.articles))

        article_scores = article_scores.cpu().detach().numpy()

        article_indices = self.calibration(
            article_scores,
            candidate_articles.articles,
            normalized_topic_prefs,
            self.theta,
            topk=self.num_slots,
        )

        return ArticleSet(articles=[candidate_articles.articles[int(idx)] for idx in article_indices])

    def compute_topic_dist(self, interest_profile):
        topic_preferences: dict[str, int] = defaultdict(int)

        for interest in interest_profile.onboarding_topics:
            topic_preferences[interest.entity_name] = max(interest.preference - 1, 0)

        if interest_profile.click_topic_counts:
            for topic, click_count in interest_profile.click_topic_counts.items():
                topic_preferences[topic] += click_count

        normalized_topic_prefs = normalized_topic_count(topic_preferences)
        return normalized_topic_prefs

    def normalized_categories_with_candidate(self, rec_topics, article):
        rec_topics_with_candidate = rec_topics.copy()
        self.add_article_to_topics(rec_topics_with_candidate, article)
        return normalized_topic_count(rec_topics_with_candidate)

    def add_article_to_categories(rec_topics, article):
        topics = extract_general_topics(article)
        for topic in topics:
            rec_topics[topic] = rec_topics.get(topic, 0) + 1

    def calibration(self, relevance_scores, articles, topic_preferences, theta, topk) -> list[Article]:
        # MR_i = \theta * reward_i - (1 - \theta)*C(S + i) # C is calibration
        # R is all candidates (not selected yet)

        recommendations = []  # final recommendation (topk index)
        rec_topics = defaultdict(int)  # frequency distribution of topics of S

        for k in range(topk):
            candidate = None  # next item
            best_candidate_score = float("-inf")

            for article_idx, article_score in enumerate(relevance_scores):  # iterate R for next item
                if article_idx in recommendations:
                    continue

                normalized_candidate_topics = self.normalized_topics_with_candidate(rec_topics, articles[article_idx])
                calibration = compute_kl_divergence(topic_preferences, normalized_candidate_topics)

                adjusted_candidate_score = (1 - theta) * article_score - (theta * calibration)
                if adjusted_candidate_score > best_candidate_score:
                    best_candidate_score = adjusted_candidate_score
                    candidate = article_idx

            if candidate is not None:
                recommendations.append(candidate)
                self.add_article_to_topics(rec_topics, articles[candidate])

        return recommendations
