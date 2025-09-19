# Simulates a request to the recommender without requiring Serverless
import warnings

from poprox_concepts.api.recommendations.v2 import RecommendationRequestV2, RecommendationResponseV2
from poprox_recommender.api.main import root
from poprox_recommender.paths import project_root
from poprox_recommender.topics import extract_general_topics

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    with open(project_root() / "tests/request_data/onboarding.json", "r") as req_file:
        raw_json = req_file.read()
        req = RecommendationRequestV2.model_validate_json(raw_json)

    event_nrms = {
        "body": raw_json,
        "queryStringParameters": {"pipeline": "nrms"},
        "isBase64Encoded": False,
    }
    event_topic_score = {
        "body": raw_json,
        "queryStringParameters": {"pipeline": "nrms_topic_scores"},
        "isBase64Encoded": False,
    }
    event_feedback_score = {
        "body": raw_json,
        "queryStringParameters": {"pipeline": "nrms_article_feedback"},
        "isBase64Encoded": False,
    }

    response_nrms = root(req.model_dump(), pipeline="nrms")
    response_nrms = RecommendationResponseV2.model_validate(response_nrms)

    response_topic_score = root(req.model_dump(), pipeline="nrms_topic_scores")
    response_topic_score = RecommendationResponseV2.model_validate(response_topic_score)

    response_feedback_score = root(req.model_dump(), pipeline="nrms_article_feedback")
    response_feedback_score = RecommendationResponseV2.model_validate(response_feedback_score)

    print("\n")
    print(f"{event_nrms['queryStringParameters']['pipeline']}")

    for idx, article in enumerate(response_nrms.recommendations.articles):
        article_topics = extract_general_topics(article)
        print(f"{idx + 1}. {article.headline} {article_topics}")

    print("\n")
    print(f"{event_topic_score['queryStringParameters']['pipeline']}")

    for idx, article in enumerate(response_topic_score.recommendations.articles):
        article_topics = extract_general_topics(article)
        print(f"{idx + 1}. {article.headline} {article_topics}")

    print("\n")
    print(f"{event_feedback_score['queryStringParameters']['pipeline']}")

    for idx, article in enumerate(response_feedback_score.recommendations.articles):
        article_topics = extract_general_topics(article)
        print(f"{idx + 1}. {article.headline} {article_topics}")
