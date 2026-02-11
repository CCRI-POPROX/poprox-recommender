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
from uuid import UUID, uuid4

import requests
from pydantic import ValidationError
from pytest import fixture, skip

from poprox_concepts.api.recommendations.v5 import RecommendationRequestV5, RecommendationResponseV5
from poprox_concepts.domain import (
    AccountInterest,
    ArticlePackage,
    CandidateSet,
    Click,
    Entity,
    InterestProfile,
    Mention,
)
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

    def request(self, req: RecommendationRequestV5 | str, pipeline: str) -> RecommendationResponseV5:
        """
        Request recommendations from the recommender.
        """
        raise NotImplementedError()


class InProcessTestService:
    """
    Test service that directly runs the request handler in-process.
    """

    def request(
        self, req: RecommendationRequestV5 | str, pipeline: str, compress: bool = False
    ) -> RecommendationResponseV5:
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

        return RecommendationResponseV5.model_validate_json(res["body"])


class DockerTestService:
    """
    Test service that directly runs the request handler in-process.
    """

    url: str

    def __init__(self, url: str):
        self.url = url

    def request(
        self, req: RecommendationRequestV5 | str, pipeline: str, compress: bool = False
    ) -> RecommendationResponseV5:
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

        logger.info("sending request to %s", self.url)
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

        return RecommendationResponseV5.model_validate_json(res_data["body"])


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
        self.article_packages = []

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
        entity_ids = {
            "General News": UUID("72bb7674-7bde-4f3e-a351-ccdeae888502"),
            "Science": UUID("1e813fd6-0998-43fb-9839-75fa96b69b32"),
            "Technology": UUID("606afcb8-3fc1-47a7-9da7-3d95115373a3"),
            "Sports": UUID("f984b26b-4333-42b3-a463-bc232bf95d5f"),
            "Oddities": UUID("16323227-4b42-4363-b67c-fd2be57c9aa1"),
            "U.S. News": UUID("66ba9689-3ad7-4626-9d20-03930d88e302"),
            "World News": UUID("45770171-36d1-4568-a270-bf80d6fe18e7"),
            "Business": UUID("5f6de24a-9a1b-4863-ab01-1ecacf4c54b7"),
            "Health": UUID("b967a4f4-ac9d-4c09-81d3-af228f846d06"),
            "Entertainment": UUID("4554dcf2-6472-43a3-bfd6-e904a2936315"),
        }
        self.added_topics = [
            AccountInterest(
                account_id=self.profile_id,
                entity_id=entity_ids[topic],
                entity_name=topic,
                entity_type="topic",
                preference=random.randint(1, 5),
                frequency=None,
            )
            for topic in topics
        ]

        # an article pacakge per topic
        all_articles = [a.article_id for a in self.candidate_articles]  # gives uuids
        self.article_packages = []
        for interest in self.added_topics:
            seed_entity = Entity(
                entity_id=interest.entity_id,
                name=interest.entity_name,
                entity_type="topic",
                source="MIND",
            )
            package_article_ids = random.sample(all_articles, min(5, len(all_articles)))
            package = ArticlePackage(
                title=f"{interest.entity_name}",
                source="MIND",
                seed=seed_entity,
                article_ids=package_article_ids,
            )
            self.article_packages.append(package)

        for article in self.candidate_articles:
            num_topics = random.randint(1, 4)
            article_topics = random.sample(list(entity_ids.keys()), num_topics)
            for topic in article_topics:
                article.mentions.append(
                    Mention(
                        source="MIND",
                        entity=Entity(
                            entity_id=entity_ids[topic],
                            name=topic,
                            entity_type="topic",
                            source="MIND",
                        ),
                        relevance=99.0,
                    )
                )

    def add_candidates(self, num_candidates):
        all_articles = self.mind_data.list_articles()
        selected_candidates = random.sample(all_articles, num_candidates)

        self.candidate_articles = [self.mind_data.lookup_article(uuid=article_id) for article_id in selected_candidates]

    def get_request(self) -> RecommendationRequestV5:
        interest_profile = InterestProfile(
            profile_id=self.profile_id,
            click_history=self.clicks,
            entity_interests=self.added_topics,
        )

        try:
            request = RecommendationRequestV5(
                candidates=CandidateSet(articles=self.candidate_articles),
                interacted=CandidateSet(articles=self.interacted_articles),
                interest_profile=interest_profile,
                num_recs=self.num_recs,
                article_packages=self.article_packages,
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
