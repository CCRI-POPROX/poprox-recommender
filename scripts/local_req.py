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

    event_nrms_with_ate = {
        "body": raw_json,
        "queryStringParameters": {"pipeline": "fm_nrms"},
        "isBase64Encoded": False,
    }

    response_nrms = root(req.model_dump(), pipeline="fm_nrms")
    response_nrms = RecommendationResponseV2.model_validate(response_nrms)

    for idx, (article, score) in enumerate(
        zip(response_nrms.recommendations.articles, response_nrms.recommendations.scores)
    ):
        article_topics = extract_general_topics(article)
        print(f"{idx + 1}. [{score:.5f}] {article.headline} {article_topics}")
