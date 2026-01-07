# Simulates a request to the recommender without requiring Serverless
import warnings

from poprox_concepts.api.recommendations.v4 import RecommendationRequestV4, RecommendationResponseV4
from poprox_recommender.api.main import root
from poprox_recommender.paths import project_root
from poprox_recommender.topics import extract_general_topics

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    with open(project_root() / "tests/request_data/onboarding.json", "r") as req_file:
        raw_json = req_file.read()
        req = RecommendationRequestV4.model_validate_json(raw_json)

    event_nrms = {
        "body": raw_json,
        "queryStringParameters": {"pipeline": "nrms"},
        "isBase64Encoded": False,
    }
    event_miner = {
       "body": raw_json,
       "queryStringParameters": {"pipeline": "miner"},
       "isBase64Encoded": False,
    }

    response_nrms = root(req.model_dump(), pipeline="nrms")
    response_nrms = RecommendationResponseV4.model_validate(response_nrms)

    response_miner = root(req.model_dump(), pipeline="miner")
    response_miner = RecommendationResponseV2.model_validate(response_miner)

    print("\n")
    print(f"{event_nrms['queryStringParameters']['pipeline']}")

    for idx, article in enumerate([impression.article for impression in response_nrms.recommendations]):
        article_topics = extract_general_topics(article)
        print(f"{idx + 1}. {article.headline} {article_topics}")

    print("\n")
    print(f"{event_miner['queryStringParameters']['pipeline']}")

    for idx, article in enumerate(response_miner.recommendations.articles):
        article_topics = extract_general_topics(article)
        print(f"{idx + 1}. {article.headline} {article_topics}")
