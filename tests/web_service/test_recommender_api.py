"""
Test the POPROX endpoint running under Serverless Offline.
"""

import logging
import os
import sys
import warnings
from threading import Condition, Lock, Thread

import requests
from pexpect import EOF, spawn
from pytest import fail, fixture, mark, skip

from poprox_recommender.config import allow_data_test_failures
from poprox_recommender.data.mind import MindData
from poprox_recommender.recommenders import recommendation_pipelines
from poprox_recommender.request_generator import RequestGenerator

logger = logging.getLogger(__name__)
try:
    PIPELINES = recommendation_pipelines().keys()
except Exception as e:
    warnings.warn("failed to load models, did you run `dvc pull`?")
    if allow_data_test_failures():
        skip("recommendation pipelines unavailable", allow_module_level=True)
    else:
        raise e


@fixture(scope="module")
def sl_listener():
    """
    Fixture that starts and stops serverless offline to test endpoint responses.
    """

    local = os.environ.get("POPROX_LOCAL_LAMBDA", None)
    if local:
        yield
        return

    thread = ServerlessBackground()
    thread.start()
    try:
        with thread.lock:
            if thread.ready.wait(15):
                logger.info("ready for tests")
                yield
            else:
                fail("serverless timed out")
    finally:
        thread.proc.sendintr()


class ServerlessBackground(Thread):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lock = Lock()
        self.ready = Condition(self.lock)

    def run(self):
        logger.info("starting serverless")
        self.proc = spawn("npx serverless offline start", logfile=sys.stdout.buffer)
        self.proc.expect(r"Server ready:")
        logger.info("server ready")
        with self.lock:
            self.ready.notify_all()
        self.proc.expect(EOF)


@mark.serverless
@mark.parametrize("pipeline", PIPELINES)
def test_recommender_api(sl_listener, pipeline):
    """
    Test the recommender API by sending dynamically generated requests.
    """
    mind_data = MindData()
    request_generator = RequestGenerator(mind_data)
    request_generator.add_candidates(5)
    request_generator.add_clicks(num_clicks=5, num_days=4)
    request_generator.add_topics(["Sports", "Science", "Politics", "Oddities", "Religion", "Business", "World news"])
    request_generator.set_num_recs(2)

    request_data = request_generator.get_request()
    request_data = request_data.model_dump_json()

    logger.info("sending request")
    res = requests.post(f"http://localhost:3000?pipeline={pipeline}", json=request_data)
    assert res.status_code == 200
    logger.info("response: %s", res.text)

    response_data = res.json()
    assert "recommendations" in response_data, "Recommendations not found in response"
    assert len(response_data["recommendations"]) > 0, "No recommendations found in response"
    # assert isinstance(response_data["recommendations"], list), "Recommendations should be a list"
