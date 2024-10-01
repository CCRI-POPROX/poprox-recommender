from collections import defaultdict

import torch as th

from poprox_concepts import ArticleSet, InterestProfile
from poprox_recommender.components.diversifiers.calibration import Calibrator
from poprox_recommender.topics import extract_general_topics, normalized_category_count


# Topic Calibration uses MMR
# to rerank recommendations according to
# topic calibration
class TopicCalibrator(Calibrator):
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

        normalized_topic_prefs = normalized_category_count(topic_preferences)
        return normalized_topic_prefs

    def add_article_to_categories(rec_topics, article):
        topics = extract_general_topics(article)
        for topic in topics:
            rec_topics[topic] = rec_topics.get(topic, 0) + 1
