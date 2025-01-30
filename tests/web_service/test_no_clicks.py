import logging
import warnings

from pydantic import ValidationError
from pytest import mark, skip

from poprox_concepts.api.recommendations import RecommendationRequest, RecommendationResponse
from poprox_recommender.config import allow_data_test_failures
from poprox_recommender.data.mind import MindData
from poprox_recommender.recommenders import recommendation_pipelines
from poprox_recommender.testing import docker_service, request_generator, send_request_and_validate

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
def test_edge_cases(docker_service, request_generator, pipeline):  # noqa:F811
    """
    Test no-clicks case.
    """
    request_generator.add_clicks(0)
    request = request_generator.get_request()

    send_request_and_validate(docker_service, request, pipeline)
    logger.info("No clicks case passed validation")
