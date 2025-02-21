# Simulates a request to the recommender without requiring Serverless
import warnings

from poprox_concepts.api.recommendations import RecommendationResponse
from poprox_recommender.handler import generate_recs
from poprox_recommender.paths import project_root
from poprox_recommender.topics import extract_general_topics

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    with open(project_root() / "tests/request_data/onboarding.json", "r") as req_file:
        raw_json = req_file.read()

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

    response_nrms = generate_recs(event_nrms, {})
    response_nrms = RecommendationResponse.model_validate_json(response_nrms["body"])

    response_static = generate_recs(event_static, {})
    response_static = RecommendationResponse.model_validate_json(response_static["body"])

    response_rrf_static_user = generate_recs(event_rrf_static_user, {})
    response_rrf_static_user = RecommendationResponse.model_validate_json(response_rrf_static_user["body"])

    for profile_id, recs in response_nrms.recommendations.items():
        print("\n")
        print(f"Recs for {profile_id}:")
        print(f"{event_nrms['queryStringParameters']['pipeline']}")

        for idx, article in enumerate(recs):
            article_topics = extract_general_topics(article)
            print(f"{idx + 1}. {article.headline} {article_topics}")

    for profile_id, recs in response_static.recommendations.items():
        print("\n")
        print(f"{event_static['queryStringParameters']['pipeline']}")

        for idx, article in enumerate(recs):
            article_topics = extract_general_topics(article)
            print(f"{idx + 1}. {article.headline} {article_topics}")

    for profile_id, recs in response_rrf_static_user.recommendations.items():
        print("\n")
        print(f"{event_rrf_static_user['queryStringParameters']['pipeline']}")

        for idx, article in enumerate(recs):
            article_topics = extract_general_topics(article)
            print(f"{idx + 1}. {article.headline} {article_topics}")
