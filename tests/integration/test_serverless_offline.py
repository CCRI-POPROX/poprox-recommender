"""
Test the POPROX endpoint running under Serverless Offline.
"""

import logging
import os
import sys
from threading import Condition, Lock, Thread

import requests
from pexpect import EOF, spawn
from pytest import fail, fixture, mark, skip

from poprox_recommender.paths import project_root
from poprox_recommender.recommenders import recommendation_pipelines

logger = logging.getLogger(__name__)
try:
    PIPELINES = recommendation_pipelines().keys()
except Exception as e:
    if "CI" not in os.environ:
        skip("recommendation pipelines unavailable")
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
def test_basic_request(sl_listener, pipeline):
    test_dir = project_root() / "tests"
    req_f = test_dir / "request_data" / "medium_request.json"
    req_body = req_f.read_text()

    logger.info("sending request")
    res = requests.post(f"http://localhost:3000?pipeline={pipeline}", req_body)
    assert res.status_code == 200
    logger.info("response: %s", res.text)
    body = res.json()
    assert "recommendations" in body
    assert len(body["recommendations"]) > 0
