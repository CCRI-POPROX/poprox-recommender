import logging

from pytest import skip

from poprox_recommender.config import allow_data_test_failures
from poprox_recommender.paths import project_root
from poprox_recommender.testing import auto_service as service  # noqa: F401

logger = logging.getLogger(__name__)


def test_compressed_request(service):  # noqa: F811
    try:
        test_dir = project_root() / "tests"
        req_f = test_dir / "request_data" / "request_body.json"
        req_body = req_f.read_text()

        logger.info("sending request")
        response = service.request(req_body, "nrms", compress=True)
        logger.info("response: %s", response.model_dump_json(indent=2))
    except FileNotFoundError as e:
        if allow_data_test_failures():
            skip("recommendation request data", allow_module_level=True)
        else:
            raise e
