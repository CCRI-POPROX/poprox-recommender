"""
Test basic request by initializing test data.
"""

import logging
import warnings

from pydantic import ValidationError
from pytest import mark, skip

from poprox_concepts.api.recommendations import RecommendationRequest, RecommendationResponse
from poprox_recommender.config import allow_data_test_failures
from poprox_recommender.recommenders import recommendation_pipelines
from poprox_recommender.request_generator import RequestGenerator
from poprox_recommender.testing import auto_service as service
from poprox_recommender.testing import mind_data

logger = logging.getLogger(__name__)
try:
    PIPELINES = recommendation_pipelines().keys()
except Exception as e:
    warnings.warn("failed to load models, did you run `dvc pull`?")
    if allow_data_test_failures():
        skip("recommendation pipelines unavailable", allow_module_level=True)
    else:
        raise e


@mark.docker
@mark.parametrize("pipeline", PIPELINES)
def test_basic_request(service, mind_data, pipeline):  # noqa: F811
    """
    Initialize request data
    """
    request_generator = RequestGenerator(mind_data())
    request_generator.add_candidates(100)
    request_generator.add_clicks(num_clicks=37, num_days=7)
    request_generator.add_topics(["Science", "Technology", "Sports", "Lifestyle", "Oddities"])
    request_generator.set_num_recs(10)
    req_body = request_generator.get_request()

    logger.info("sending request")
    response = service.request(req_body, pipeline)
    logger.info("response: %s", response.model_dump_json(indent=2))
    assert response.recommendations
