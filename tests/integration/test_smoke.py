"""
Test the POPROX API through direct call.
"""

import logging

from pytest import xfail

from poprox_concepts.api.recommendations.v4 import RecommendationRequestV4
from poprox_recommender.config import allow_data_test_failures
from poprox_recommender.paths import project_root
from poprox_recommender.recommenders import PipelineLoadError, select_articles

logger = logging.getLogger(__name__)


def test_direct_basic_request():
    test_dir = project_root() / "tests"
    req_f = test_dir / "request_data" / "basic-request.json"
    try:
        req = RecommendationRequestV4.model_validate_json(req_f.read_text())
    except FileNotFoundError as e:
        if allow_data_test_failures():
            xfail("data not pulled")
        else:
            raise e

    logger.info("generating recommendations")
    try:
        recs, _ = select_articles(
            req.candidates,
            req.interacted,
            req.interest_profile,
        )
    except PipelineLoadError as e:
        if allow_data_test_failures():
            xfail("models not pulled")
        else:
            raise e

    # do we get recommendations?
    assert len(recs.impressions) > 0


def test_direct_basic_request_without_clicks():
    test_dir = project_root() / "tests"
    req_f = test_dir / "request_data" / "basic-request.json"
    try:
        req = RecommendationRequestV4.model_validate_json(req_f.read_text())
    except FileNotFoundError as e:
        if allow_data_test_failures():
            xfail("models not pulled")
        else:
            raise e

    logger.info("generating recommendations")

    profile = req.interest_profile
    profile.click_history = []
    try:
        recs, _ = select_articles(
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
    assert len(recs.impressions) > 0


def test_direct_basic_request_explicit_none():
    test_dir = project_root() / "tests"
    req_f = test_dir / "request_data" / "basic-request.json"
    try:
        req = RecommendationRequestV4.model_validate_json(req_f.read_text())
    except FileNotFoundError as e:
        if allow_data_test_failures():
            xfail("data not pulled")
        else:
            raise e

    logger.info("generating recommendations")
    try:
        recs, _ = select_articles(
            req.candidates, req.interacted, req.interest_profile, pipeline_params={"pipeline": None}
        )
    except PipelineLoadError as e:
        if allow_data_test_failures():
            logger.warning("pipeline failed to load", exc_info=e)
            xfail("models not pulled")
        else:
            raise e

    # do we get recommendations?
    assert len(recs.impressions) > 0
