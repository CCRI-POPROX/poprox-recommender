"""
Test the POPROX endpoint running under Serverless Offline.
"""

import logging
import warnings

from pytest import mark, skip

from poprox_recommender.config import allow_data_test_failures
from poprox_recommender.paths import project_root
from poprox_recommender.recommenders import recommendation_pipelines
from poprox_recommender.testing import auto_service as service  # noqa: F401

logger = logging.getLogger(__name__)
try:
    PIPELINES = recommendation_pipelines().keys()
except Exception as e:
    warnings.warn("failed to load models, did you run `dvc pull`?")
    if allow_data_test_failures():
        skip("recommendation pipelines unavailable", allow_module_level=True)
    else:
        raise e


@mark.parametrize("pipeline", PIPELINES)
def test_basic_request(service, pipeline):  # noqa: F811
    test_dir = project_root() / "tests"
    req_f = test_dir / "request_data" / "request_body.json"
    req_body = req_f.read_text()

    logger.info("sending request")
    response = service.request(req_body, pipeline)
    logger.info("response: %s", response.model_dump_json(indent=2))
    assert response.recommendations
