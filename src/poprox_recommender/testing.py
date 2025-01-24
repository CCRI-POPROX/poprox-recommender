"""
Support code for testing the poprox-recommender components and service.

This lives in the main package so it can be easily imported from any of the
tests, regardless of their subdirectories.
"""

from collections.abc import Generator
from typing import Protocol

from pytest import fixture

from poprox_concepts.api.recommendations import RecommendationRequest, RecommendationResponse


class TestService(Protocol):
    """
    Interface for test services bridges that accept requests and return
    responses. It abstracts direct local runs, Docker, and AWS Lambda test
    endpoints.
    """

    def request(self, req: RecommendationRequest) -> RecommendationResponse:
        """
        Request recommendations from the recommender.
        """
        raise NotImplementedError()


class InProcessTestService:
    """
    Test service that directly runs the request handler in-process.
    """

    def request(self, req: RecommendationRequest | str) -> RecommendationResponse:
        # defer to here so we don't always import the handler
        from poprox_recommender.handler import generate_recs

        if not isinstance(req, str):
            req = req.model_dump_json()

        event = {
            "body": req,
            "queryStringParameters": {"pipeline": "nrms"},
            "isBase64Encoded": False,
        }
        res = generate_recs(event, {})
        return RecommendationResponse.model_validate_json(res["body"])


@fixture
def local_service() -> Generator[InProcessTestService, None, None]:
    yield InProcessTestService()
