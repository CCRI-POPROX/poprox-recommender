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
    event_static = {
        "body": raw_json,
        "queryStringParameters": {"pipeline": "nrms-topics-static"},
        "isBase64Encoded": False,
    }
    event_rrf_static_user = {
        "body": raw_json,
        "queryStringParameters": {"pipeline": "nrms_rrf_static_user"},
        "isBase64Encoded": False,
    }

    response_nrms = root(req.model_dump(), pipeline="nrms")
    response_nrms = RecommendationResponseV2.model_validate(response_nrms)

    response_static = root(req.model_dump(), pipeline="nrms-topics-static")
    response_static = RecommendationResponseV2.model_validate(response_static)

    response_rrf_static_user = root(req.model_dump(), pipeline="nrms_rrf_static_user")
    response_rrf_static_user = RecommendationResponseV2.model_validate(response_rrf_static_user)

    print("\n")
    print(f"{event_nrms['queryStringParameters']['pipeline']}")

    for idx, article in enumerate(response_nrms.recommendations.articles):
        article_topics = extract_general_topics(article)
        print(f"{idx + 1}. {article.headline} {article_topics}")

    print("\n")
    print(f"{event_static['queryStringParameters']['pipeline']}")

    for idx, article in enumerate(response_static.recommendations.articles):
        article_topics = extract_general_topics(article)
        print(f"{idx + 1}. {article.headline} {article_topics}")

    print("\n")
    print(f"{event_rrf_static_user['queryStringParameters']['pipeline']}")

    for idx, article in enumerate(response_rrf_static_user.recommendations.articles):
        article_topics = extract_general_topics(article)
        print(f"{idx + 1}. {article.headline} {article_topics}")
