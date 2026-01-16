"""
Test the POPROX endpoint running under Serverless Offline.
"""

import logging
import warnings

from pytest import mark, skip, xfail

from poprox_recommender.config import allow_data_test_failures
from poprox_recommender.paths import project_root
from poprox_recommender.recommenders import discover_pipelines
from poprox_recommender.testing import auto_service as service  # noqa: F401

logger = logging.getLogger(__name__)


@mark.parametrize("pipeline", discover_pipelines())
def test_basic_request(service, pipeline):  # noqa: F811
    test_dir = project_root() / "tests"
    req_f = test_dir / "request_data" / "request_body.json"
    try:
        req_body = req_f.read_text()
    except FileNotFoundError as e:
        if allow_data_test_failures():
            xfail("data not pulled")
        else:
            raise e

    logger.info("sending request")
    response = service.request(req_body, pipeline, compress=True)
    logger.info("response: %s", response.model_dump_json(indent=2))
    assert response.recommendations
