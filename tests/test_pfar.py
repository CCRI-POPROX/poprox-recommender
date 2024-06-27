import json
import logging
import random
from pathlib import Path

from poprox_concepts import Article, ClickHistory, InterestProfile
from poprox_recommender.default import select_articles, user_topic_preference
from poprox_recommender.topics import GENERAL_TOPICS, extract_general_topics, match_news_topics_to_general

logger = logging.getLogger(__name__)


def load_test_articles():
    event_path = Path(__file__).parent / "request_body.json"
    with open(event_path, "r") as j:
        req_body = json.loads(j.read())

        todays_articles = [Article.model_validate(attrs) for attrs in req_body["todays_articles"]]

        past_articles = [Article.model_validate(attrs) for attrs in req_body["past_articles"]]

        click_history = [ClickHistory.model_validate(attrs) for attrs in req_body["click_data"]]

        num_recs = req_body["num_recs"]

    return todays_articles, past_articles, click_history, num_recs


def test_topic_classification():
    todays_articles, _, _, _ = load_test_articles()
    topic_matched_dict, todays_article_matched_topics = match_news_topics_to_general(todays_articles)
    assert len(todays_article_matched_topics.keys()) > 0

    random_10_topic = random.sample(list(topic_matched_dict.keys()), 10)
    for article_topic in random_10_topic:
        assert len(topic_matched_dict[article_topic]) > 0


def test_extract_generalized_topic():
    todays_articles, _, _, _ = load_test_articles()
    for article in todays_articles:
        generalized_topics = extract_general_topics(article)
        for topic in generalized_topics:
            assert topic in GENERAL_TOPICS


def test_user_topic_pref():
    todays_articles, past_articles, click_data, num_recs = load_test_articles()
    interest_profile = InterestProfile.model_validate(
        {
            "click_history": click_data[0],
            "onboarding_topics": [],
        }
    )

    user_preference_dict = user_topic_preference(past_articles, click_data[0])
    algo_params = {"user_topic_preference": user_preference_dict}
    recommendations = select_articles(todays_articles, past_articles, interest_profile, num_recs, algo_params)
    assert len(recommendations) > 0
