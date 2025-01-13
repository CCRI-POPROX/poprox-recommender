"""
Test recommendation format
"""

import logging

from pydantic import ValidationError
from pytest import skip

from poprox_concepts import ArticleSet
from poprox_concepts.api.recommendations import RecommendationRequest
from poprox_recommender.config import allow_data_test_failures
from poprox_recommender.paths import project_root

logger = logging.getLogger(__name__)


def test_recommendation_format():
    test_dir = project_root() / "tests"
    req_f = test_dir / "request_data" / "request_body.json"
    if allow_data_test_failures() and not req_f.exists():
        skip("Request file does not exist")

    try:
        req = RecommendationRequest.model_validate_json(req_f.read_text())
    except ValidationError as e:
        raise e

    todays_articles = req.todays_articles
    try:
        ArticleSet(articles=todays_articles)
    except ValidationError as e:
        raise e

    past_articles = req.past_articles
    try:
        ArticleSet(articles=past_articles)
    except ValidationError as e:
        raise e

    logger.info("All articles passed validation.")
