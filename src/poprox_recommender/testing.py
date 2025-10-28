"""
Support code for testing the poprox-recommender components and service.

This lives in the main package so it can be easily imported from any of the
tests, regardless of their subdirectories.
"""

import base64
import gzip
import json
import logging
import os
import random
import subprocess as sp
from collections.abc import Generator
from datetime import datetime, timedelta
from signal import SIGINT
from time import sleep
from typing import Any, List, Protocol
from uuid import uuid4

import requests
from pydantic import ValidationError
from pytest import fixture, skip

from poprox_concepts.api.recommendations.v4 import RecommendationRequestV4, RecommendationResponseV4
from poprox_concepts.domain import AccountInterest, CandidateSet, Click, InterestProfile
from poprox_recommender.config import allow_data_test_failures
from poprox_recommender.data.mind import MindData
from poprox_recommender.recommenders.load import PipelineLoadError

logger = logging.getLogger(__name__)


class TestService(Protocol):
    """
    Interface for test services bridges that accept requests and return
    responses. It abstracts direct local runs, Docker, and AWS Lambda test
    endpoints.
    """

    def request(self, req: RecommendationRequestV4 | str, pipeline: str) -> RecommendationResponseV4:
        """
        Request recommendations from the recommender.
        """
        raise NotImplementedError()


class InProcessTestService:
    """
    Test service that directly runs the request handler in-process.
    """

    def request(
        self, req: RecommendationRequestV4 | str, pipeline: str, compress: bool = False
    ) -> RecommendationResponseV4:
        # defer to here so we don't always import the handler
        from poprox_recommender.api.main import handler

        if not isinstance(req, str):
            req = req.model_dump_json()

        req_txt = json.dumps(json.loads(req))

        event: dict[str, Any] = {
            "resource": "/",
            "path": "/",
            "httpMethod": "POST",
            "requestContext": {},
            "headers": {},
            "queryStringParameters": {"pipeline": pipeline},
            "body": req_txt,
            "isBase64Encoded": False,
        }

        if compress:
            event.update(
                {
                    "headers": {"Content-Type": "application/json", "Content-Encoding": "gzip"},
                    "body": base64.encodebytes(gzip.compress(req_txt.encode())).decode("ascii"),
                    "isBase64Encoded": True,
                }
            )

        try:
            res = handler(event, {})
        except PipelineLoadError as e:
            logger.error("error loading pipeline %s", pipeline, exc_info=e)
            if allow_data_test_failures():
                skip("recommendation pipelines unavailable", allow_module_level=True)
            else:
                raise e

        return RecommendationResponseV4.model_validate_json(res["body"])


class DockerTestService:
    """
    Test service that directly runs the request handler in-process.
    """

    url: str

    def __init__(self, url: str):
        self.url = url

    def request(
        self, req: RecommendationRequestV4 | str, pipeline: str, compress: bool = False
    ) -> RecommendationResponseV4:
        if not isinstance(req, str):
            req = req.model_dump_json()

        req_txt = json.dumps(json.loads(req))

        event: dict[str, Any] = {
            "resource": "/",
            "path": "/",
            "httpMethod": "POST",
            "requestContext": {},
            "headers": {},
            "queryStringParameters": {"pipeline": pipeline},
            "body": req_txt,
            "isBase64Encoded": False,
        }

        if compress:
            event.update(
                {
                    "headers": {"Content-Type": "application/json", "Content-Encoding": "gzip"},
                    "body": base64.encodebytes(gzip.compress(req_txt.encode())).decode("ascii"),
                    "isBase64Encoded": True,
                }
            )

        result = requests.post(self.url, json=event)
        res_data = result.json()
        if result.status_code != 200:
            logger.error("Lambda function failed with type %s", res_data["errorType"])
            logger.info("Lambda error message: %s", res_data["errorMessage"])
            if res_data["stackTrace"]:
                logger.info("Stack trace:%s", [f"\n  {i}: {t}" for (i, t) in enumerate(res_data["stackTrace"], 1)])
            raise AssertionError("lambda function failed")

        if res_data["statusCode"] != 200:
            logger.error("HTTP succeeded but lambda failed with code %s", res_data["statusCode"])
            logger.info("result: %s", json.dumps(res_data, indent=2))
            raise AssertionError("lambda request failed")

        return RecommendationResponseV4.model_validate_json(res_data["body"])


def local_service_impl() -> Generator[TestService, None, None]:
    yield InProcessTestService()


local_service = fixture(local_service_impl)


def docker_service_impl() -> Generator[TestService, None, None]:
    logger.info("building docker container")
    sp.check_call(["docker-buildx", "build", "-t", "poprox-recommender:test", "."])
    logger.info("starting docker container")
    proc = sp.Popen(["docker", "run", "--rm", "-p", "18080:8080", "poprox-recommender:test"])
    sleep(1)
    try:
        yield DockerTestService("http://localhost:18080/2015-03-31/functions/function/invocations")
    finally:
        logger.info("stopping docker container")
        proc.send_signal(SIGINT)
        proc.wait()


docker_service = fixture(scope="session")(docker_service_impl)


@fixture(scope="session")
def auto_service() -> Generator[TestService, None, None]:
    backend = os.environ.get("POPROX_TEST_TARGET", "local")
    if backend == "local":
        yield from local_service_impl()
    elif backend == "docker-auto":
        yield from docker_service_impl()
    elif backend == "docker":
        # use already-running docker
        port = os.environ.get("POPROX_TEST_PORT", "9000")
        yield DockerTestService(f"http://localhost:{port}/2015-03-31/functions/function/invocations")


class RequestGenerator:
    """
    Class to generate recommendation request using click history,
    onboarding topics, and candidate articles from MIND
    """

    def __init__(self, mind_data: MindData):
        self.mind_data = mind_data
        self.profile_id = uuid4()
        self.candidate_articles = list()
        self.interacted_articles = list()
        self.added_topics = list()
        self.clicks = list()

    def set_num_recs(self, num_recs: int):
        self.num_recs = num_recs

    def add_clicks(self, num_clicks: int, num_days: int | None = None):
        all_articles = self.mind_data.list_articles()

        if num_days:
            start_date = datetime.now() - timedelta(days=num_days - 1)
            timestamps = [start_date + timedelta(days=random.randint(0, num_days - 1)) for _ in range(num_clicks)]
            random.shuffle(timestamps)
        else:
            timestamps = [datetime.now()] * num_clicks
        # generate click history
        self.clicks = [
            Click(
                article_id=random.choice(all_articles),
                newsletter_id=uuid4(),
                timestamp=timestamps[i],
            )
            for i in range(num_clicks)
        ]

        self.interacted_articles = [self.mind_data.lookup_article(uuid=click.article_id) for click in self.clicks]

    def add_topics(self, topics: List[str]):
        self.added_topics = [
            AccountInterest(
                account_id=self.profile_id,
                entity_id=uuid4(),
                entity_name=topic,
                preference=random.randint(1, 5),
                frequency=None,
            )
            for topic in topics
        ]

    def add_candidates(self, num_candidates):
        all_articles = self.mind_data.list_articles()
        selected_candidates = random.sample(all_articles, num_candidates)

        self.candidate_articles = [self.mind_data.lookup_article(uuid=article_id) for article_id in selected_candidates]

    def get_request(self) -> RecommendationRequestV4:
        interest_profile = InterestProfile(
            profile_id=self.profile_id,
            click_history=self.clicks,
            onboarding_topics=self.added_topics,
        )

        try:
            request = RecommendationRequestV4(
                candidates=CandidateSet(articles=self.candidate_articles),
                interacted=CandidateSet(articles=self.interacted_articles),
                interest_profile=interest_profile,
                num_recs=self.num_recs,
            )
            return request
        except ValidationError as e:
            raise ValueError(f"Generated request is invalid: {e}")


@fixture(scope="session")
def mind_data():
    try:
        yield MindData()
    except FileNotFoundError as e:
        if allow_data_test_failures("mind"):
            skip("MIND data not available")
        else:
            raise e
