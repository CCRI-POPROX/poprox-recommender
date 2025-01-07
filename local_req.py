# Simulates a request to the recommender without requiring Serverless
import warnings

from poprox_concepts.api.recommendations import RecommendationResponse
from poprox_recommender.handler import generate_recs
from poprox_recommender.topics import extract_general_topics

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    with open("tests/request_data/no_clicks.json", "r") as req_file:
        raw_json = req_file.read()

    event_1 = {
        "body": raw_json,
        "queryStringParameters": {"pipeline": "nrms-topics-candidate"},
        "isBase64Encoded": False,
    }
    event_2 = {"body": raw_json, "queryStringParameters": {"pipeline": "nrms-topics-clicked"}, "isBase64Encoded": False}
    event_3 = {"body": raw_json, "queryStringParameters": {"pipeline": "nrms-topics-static"}, "isBase64Encoded": False}

    response_1 = generate_recs(event_1, {})
    response_1 = RecommendationResponse.model_validate_json(response_1["body"])

    response_2 = generate_recs(event_2, {})
    response_2 = RecommendationResponse.model_validate_json(response_2["body"])

    response_3 = generate_recs(event_3, {})
    response_3 = RecommendationResponse.model_validate_json(response_3["body"])

    for profile_id, recs in response_1.recommendations.items():
        print("\n")
        print(f"Recs for {profile_id}:")
        print(f"{event_1['queryStringParameters']['pipeline']}")

        for idx, article in enumerate(recs):
            article_topics = extract_general_topics(article)
            print(f"{idx+1}. {article.headline} {article_topics}")

    for profile_id, recs in response_2.recommendations.items():
        print("\n")
        print(f"{event_2['queryStringParameters']['pipeline']}")

        for idx, article in enumerate(recs):
            article_topics = extract_general_topics(article)
            print(f"{idx+1}. {article.headline} {article_topics}")

    for profile_id, recs in response_3.recommendations.items():
        print("\n")
        print(f"{event_3['queryStringParameters']['pipeline']}")

        for idx, article in enumerate(recs):
            article_topics = extract_general_topics(article)
            print(f"{idx+1}. {article.headline} {article_topics}")
