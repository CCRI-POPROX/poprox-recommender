"""
Test the POPROX API through direct call.
"""

import logging

from pytest import xfail

from poprox_concepts import CandidateSet, Click
from poprox_concepts.api.recommendations import RecommendationRequest
from poprox_recommender.config import allow_data_test_failures
from poprox_recommender.paths import project_root
from poprox_recommender.recommenders import PipelineLoadError, select_articles

logger = logging.getLogger(__name__)


def test_direct_basic_request():
    test_dir = project_root() / "tests"
    req_f = test_dir / "request_data" / "basic-request.json"
    req = RecommendationRequest.model_validate_json(req_f.read_text())

    logger.info("generating recommendations")
    try:
        outputs = select_articles(
            req.candidates,
            req.interacted,
            req.interest_profile,
        )
    except PipelineLoadError as e:
        if allow_data_test_failures():
            xfail("data not pulled")
        else:
            raise e

    # do we get recommendations?
    assert len(outputs.default.articles) > 0


def test_direct_basic_request_without_clicks():
    test_dir = project_root() / "tests"
    req_f = test_dir / "request_data" / "basic-request.json"
    req = RecommendationRequest.model_validate_json(req_f.read_text())

    logger.info("generating recommendations")

    profile = req.interest_profile
    profile.click_history = []
    try:
        outputs = select_articles(
            req.candidates,
            req.interacted,
            req.interest_profile,
        )
    except PipelineLoadError as e:
        if allow_data_test_failures():
            xfail("data not pulled")
        else:
            raise e

    # do we get recommendations?
    assert len(outputs.default.articles) > 0
