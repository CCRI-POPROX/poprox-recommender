import logging
import os
import sys
from threading import Condition, Lock, Thread

from pexpect import EOF, spawn
from pytest import fail, fixture

logger = logging.getLogger(__name__)


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
