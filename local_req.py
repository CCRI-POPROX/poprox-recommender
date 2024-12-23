# Simulates a request to the recommender without requiring Serverless
import warnings

from poprox_concepts.api.recommendations import RecommendationResponse
from poprox_recommender.handler import generate_recs
from poprox_recommender.topics import extract_general_topics

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    with open("tests/request_data/no_clicks.json", "r") as req_file:
        raw_json = req_file.read()

    event = {"body": raw_json, "queryStringParameters": {"pipeline": "nrms-topics"}, "isBase64Encoded": False}
    response = generate_recs(event, {})
    response = RecommendationResponse.model_validate_json(response["body"])

    for profile_id, recs in response.recommendations.items():
        print(f"Recs for {profile_id}:")

        for idx, article in enumerate(recs):
            article_topics = extract_general_topics(article)
            print(f"{idx+1}. {article.headline} {article_topics}")
