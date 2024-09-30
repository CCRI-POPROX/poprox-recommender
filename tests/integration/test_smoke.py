"""
Test the POPROX API through direct call.
"""

import logging

from pytest import skip

from poprox_concepts import ArticleSet, Click
from poprox_concepts.api.recommendations import RecommendationRequest
from poprox_recommender.paths import project_root
from poprox_recommender.recommenders import select_articles

logger = logging.getLogger(__name__)


def test_direct_basic_request():
    test_dir = project_root() / "tests"
    req_f = test_dir / "request_data" / "basic-request.json"
    req = RecommendationRequest.model_validate_json(req_f.read_text())

    logger.info("generating recommendations")
    try:
        outputs = select_articles(
            ArticleSet(articles=req.todays_articles),
            ArticleSet(articles=req.past_articles),
            req.interest_profile,
        )
    except FileNotFoundError:
        # FIXME: when we have configuration files, separate "cannot load" from "cannot run"
        skip("model data not available")
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
            ArticleSet(articles=req.todays_articles),
            ArticleSet(articles=req.past_articles),
            req.interest_profile,
        )
    except FileNotFoundError:
        # FIXME: when we have configuration files, separate "cannot load" from "cannot run"
        skip("model data not available")
    # do we get recommendations?
    assert len(outputs.default.articles) > 0
