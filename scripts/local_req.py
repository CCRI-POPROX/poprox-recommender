# Simulates a request to the recommender without requiring Serverless
import warnings

from poprox_concepts.api.recommendations.v2 import RecommendationRequestV2, RecommendationResponseV2
from poprox_recommender.api.main import root
from poprox_recommender.paths import project_root

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    with open(project_root() / "tests/request_data/onboarding.json", "r") as req_file:
        raw_json = req_file.read()
        req = RecommendationRequestV2.model_validate_json(raw_json)

    event_llm = {
        "body": raw_json,
        "queryStringParameters": {"pipeline": "llm_rank_rewrite"},
        "isBase64Encoded": False,
    }

    response_llm = root(req.model_dump(), pipeline="llm_rank_rewrite")
    response_llm = RecommendationResponseV2.model_validate(response_llm)

    print("\n")
    print(f"{event_llm['queryStringParameters']['pipeline']}")
    for idx, article in enumerate(response_llm.recommendations.articles):
        print(f"{idx + 1}. {article.headline}")
