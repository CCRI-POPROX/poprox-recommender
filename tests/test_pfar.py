import json
import sys
sys.path.append('src')
from poprox_concepts import Article
from poprox_recommender.handler import general_topics, match_news_topics_to_general, extract_general_topic

def load_test_articles():
    event_path = "/home/sun00587/research/POPROX/poprox-recommender/tests/request_body.json" # might need to config the path correctly
    with open(event_path, 'r') as j:
        req_body = json.loads(j.read())
        todays_articles = [
            Article.model_validate(attrs) for attrs in req_body["todays_articles"]
        ]
    return todays_articles

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