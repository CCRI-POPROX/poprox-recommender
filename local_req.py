# Simulates a request to the recommender without requiring Serverless
import warnings

from poprox_concepts.api.recommendations import RecommendationResponse
from poprox_recommender.handler import generate_recs
from poprox_recommender.topics import extract_general_topics

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    with open("tests/request_data/no_clicks.json", "r") as req_file:
        raw_json = req_file.read()

    event_static = {
        "body": raw_json,
        "queryStringParameters": {"pipeline": "nrms-topics-static"},
        "isBase64Encoded": False,
    }
    event_candidate = {
        "body": raw_json,
        "queryStringParameters": {"pipeline": "nrms-topics-candidate"},
        "isBase64Encoded": False,
    }
    event_clicked = {
        "body": raw_json,
        "queryStringParameters": {"pipeline": "nrms-topics-clicked"},
        "isBase64Encoded": False,
    }

    response_static = generate_recs(event_static, {})
    response_static = RecommendationResponse.model_validate_json(response_static["body"])

    response_candidate = generate_recs(event_candidate, {})
    response_candidate = RecommendationResponse.model_validate_json(response_candidate["body"])

    response_clicked = generate_recs(event_clicked, {})
    response_clicked = RecommendationResponse.model_validate_json(response_clicked["body"])

    for profile_id, recs in response_static.recommendations.items():
        print("\n")
        print(f"{event_static['queryStringParameters']['pipeline']}")

        for idx, article in enumerate(recs):
            article_topics = extract_general_topics(article)
            print(f"{idx+1}. {article.headline} {article_topics}")

    for profile_id, recs in response_candidate.recommendations.items():
        print("\n")
        print(f"Recs for {profile_id}:")
        print(f"{event_candidate['queryStringParameters']['pipeline']}")

        for idx, article in enumerate(recs):
            article_topics = extract_general_topics(article)
            print(f"{idx+1}. {article.headline} {article_topics}")

    for profile_id, recs in response_clicked.recommendations.items():
        print("\n")
        print(f"{event_clicked['queryStringParameters']['pipeline']}")

        for idx, article in enumerate(recs):
            article_topics = extract_general_topics(article)
            print(f"{idx+1}. {article.headline} {article_topics}")
