import logging
import warnings

import requests
from pydantic import ValidationError
from pytest import mark, skip

from poprox_concepts.api.recommendations import RecommendationRequest, RecommendationResponse
from poprox_recommender.config import allow_data_test_failures
from poprox_recommender.data.mind import MindData
from poprox_recommender.paths import project_root
from poprox_recommender.recommenders import recommendation_pipelines
from poprox_recommender.request_generator import RequestGenerator
from poprox_recommender.testing import sl_listener

logger = logging.getLogger(__name__)
try:
    PIPELINES = recommendation_pipelines().keys()
except Exception as e:
    warnings.warn("failed to load models, did you run `dvc pull`?")
    if allow_data_test_failures():
        skip("recommendation pipelines unavailable", allow_module_level=True)
    else:
        raise e


def send_request_and_validate(request, pipeline):
    request_data_json = request.model_dump_json()
    req = RecommendationRequest.model_validate_json(request_data_json)
    assert isinstance(req.interest_profile.click_history, list)

    logger.info("sending request to: http://localhost:3000")
    response = requests.post(f"http://localhost:3000?pipeline={pipeline}", json=request_data_json)
    assert response.status_code == 200
    logger.info("response: %s", response.text)

    response_data = response.json()
    validated_response = RecommendationResponse.model_validate(response_data)
    assert isinstance(validated_response.recommendations, dict)
    assert len(validated_response.recommendations) > 0, "No recommendations in response"

    recommended_articles = list(validated_response.recommendations.values())[0]
    assert (
        len(recommended_articles) == request_generator.num_recs
    ), f"Expected {request_generator.num_recs} recommendations, but got {recommended_articles}"


"""
Initialize a RequestGenerator object, add candidates, clicks, topics,
and set the number of recommendations.
"""
mind_data = MindData()
request_generator = RequestGenerator(mind_data)
request_generator.add_candidates(100)
request_generator.add_clicks(num_clicks=46, num_days=7)
request_generator.add_topics(["Science"])  # one onboarding topic
request_generator.set_num_recs(10)


@mark.serverless
@mark.parametrize("pipeline", PIPELINES)
def test_minimal_onboarding(sl_listener, pipeline):  # noqa:F811
    """
    Test the case when there is minimal onboarding topic.
    """
    request = request_generator.get_request()
    send_request_and_validate(request, pipeline)
    logger.info("Minimal onboarding case passed validation")
