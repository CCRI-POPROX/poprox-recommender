import json
import sys
from poprox_concepts import Article
from poprox_recommender.handler import match_news_topics_to_general

def generate_recs(event_path):
    with open(event_path, 'r') as j:
        req_body = json.loads(j.read())
        todays_articles = [
            Article.model_validate(attrs) for attrs in req_body["todays_articles"]
        ]

    topic_matched_dict, todays_article_matched_topics = match_news_topics_to_general(todays_articles)
    print(topic_matched_dict)
    for article_topic in todays_article_matched_topics:
        print(todays_article_matched_topics[article_topic])
        break

if __name__ == '__main__':
    event_path = 'request_body.json'
    generate_recs(event_path)