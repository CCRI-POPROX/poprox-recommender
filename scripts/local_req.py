# Simulates a request to the recommender without requiring Serverless
import json
import warnings

from poprox_concepts.api.recommendations.v2 import RecommendationRequestV2, RecommendationResponseV2
from poprox_recommender.api.main import root
from poprox_recommender.paths import project_root

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    project_root_path = project_root()
    profile_paths = (project_root_path / "tests" / "request_data" / "profiles").glob("*.json")
    for profile_path in profile_paths:
        print(f"Loading profile from {profile_path}")

        with open(profile_path, "r") as profile_file:
            raw_json = profile_file.read()
            req = RecommendationRequestV2.model_validate_json(raw_json)

        event_profile = {
            "body": raw_json,
            "queryStringParameters": {"pipeline": "llm_rank_rewrite"},
            "isBase64Encoded": False,
        }

        response_llm = root(req.model_dump(), pipeline="llm_rank_rewrite")
        response_llm = RecommendationResponseV2.model_validate(response_llm)

        structured_output = {
            "recommendations": [
                {
                    "rank": idx + 1,
                    "headline": article.headline,
                }
                for idx, article in enumerate(response_llm.recommendations.articles)
            ],
            "profile_name": profile_path.stem,
            "candidate_pool": [i.headline for i in req.candidates.articles],
        }

        with open(f"data/{profile_path.stem}_output.json", "w") as output_file:
            json.dump(structured_output, output_file, indent=2)
