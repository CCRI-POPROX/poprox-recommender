import json
import sys

import torch as th

sys.path.append(
    "/home/sun00587/research/POPROX/poprox-recommender/src"
)  # should not need this when model path func is ready
from safetensors.torch import load_file

from poprox_concepts import Article, ClickHistory
from poprox_recommender.default import select_articles, user_topic_preference
from poprox_recommender.topics import extract_general_topic, general_topics, match_news_topics_to_general

try:
    import pytest

    pytestmark = pytest.mark.skip("not a test module")
except ImportError:
    pass


def load_test_articles():
    event_path = "/home/sun00587/research/POPROX/poprox-recommender/tests/request_body.json"  # update when the model path func is ready  # noqa: E501
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
    print("***************** topic matched dict *****************")
    print(topic_matched_dict)
    for article_topic in todays_article_matched_topics:
        print(todays_article_matched_topics[article_topic])
        break


def test_extract_generalized_topic():
    todays_articles, _, _, _ = load_test_articles()
    for article in todays_articles:
        generalized_topics = extract_general_topic(article)
        for topic in generalized_topics:
            assert topic in general_topics


def load_model(device_name=None):
    if device_name is None:
        device_name = "cuda" if th.cuda.is_available() else "cpu"

    load_path = "/home/sun00587/research/POPROX/poprox-recommender/src/models/model.safetensors"  # update when the model path func is ready  # noqa: E501
    checkpoint = load_file(load_path)

    return checkpoint, device_name


if __name__ == "__main__":
    todays_articles, past_articles, click_data, num_recs = load_test_articles()

    user_preference_dict = user_topic_preference(past_articles, click_data)
    algo_params = {"user_topic_preference": user_preference_dict}
    recommendations = select_articles(todays_articles, past_articles, click_data, num_recs, algo_params)
    print(recommendations)
