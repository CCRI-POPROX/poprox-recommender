import logging

from pytest import mark

from poprox_recommender.paths import project_root
from poprox_recommender.testing import auto_service as service  # noqa: F401

logger = logging.getLogger(__name__)


@mark.parametrize("pipeline", ["nrms"])
def test_compressed_request(service, pipeline):  # noqa: F811
    test_dir = project_root() / "tests"
    req_f = test_dir / "request_data" / "request_body.json"
    req_body = req_f.read_text()

    logger.info("sending request")
    response = service.request(req_body, pipeline, compress=True)
    logger.info("response: %s", response.model_dump_json(indent=2))
    assert response.recommendations
