import json
import torch as th
import sys
sys.path.append('/home/XLL1713/POPROX/poprox/poprox-recommender/src')
print(sys.path)
from poprox_concepts import Article, ClickHistory
from poprox_recommender.handler import general_topics, match_news_topics_to_general, extract_general_topic
from poprox_recommender.default import select_articles, user_topic_preference
from poprox_recommender.paths import project_root, model_file_path
from safetensors.torch import load_file


def load_test_articles():
    event_path = "/home/XLL1713/POPROX/poprox/poprox-recommender/tests/request_body.json" # might need to config the path correctly
    with open(event_path, 'r') as j:
        req_body = json.loads(j.read())

        todays_articles = [Article.model_validate(attrs) for attrs in req_body["todays_articles"]]

        past_articles = [Article.model_validate(attrs) for attrs in req_body["past_articles"]]

        click_history = [ClickHistory.model_validate(attrs) for attrs in req_body["click_data"]]

        num_recs = req_body["num_recs"]

    return todays_articles, past_articles, click_history, num_recs

def test_topic_classification():
    todays_articles = load_test_articles()
    topic_matched_dict, todays_article_matched_topics = match_news_topics_to_general(todays_articles)
    print('***************** topic matched dict *****************')
    print(topic_matched_dict)
    for article_topic in todays_article_matched_topics:
        print(todays_article_matched_topics[article_topic])
        break


def test_extract_generalized_topic():
    todays_articles = load_test_articles()
    for article in todays_articles:
        generalized_topics = extract_general_topic(article)
        for topic in generalized_topics:
            assert topic in general_topics

def load_model(device_name=None):

    if device_name is None:
        device_name = "cuda" if th.cuda.is_available() else "cpu"

    load_path = model_file_path("model.safetensors")
    checkpoint = load_file(load_path)

    return checkpoint, device_name


if __name__ == '__main__':
    todays_articles, past_articles, click_data, num_recs = load_test_articles()

    MODEL, DEVICE = load_model()
    TOKEN_MAPPING = 'distilbert-base-uncased'
    user_preference_dict = user_topic_preference(past_articles, click_data)
    algo_params = {'user_topic_preference': user_preference_dict}
    recommendations = select_articles(
        todays_articles,
        past_articles,
        click_data,
        MODEL,
        DEVICE,
        TOKEN_MAPPING,
        num_recs,
        algo_params
    )
