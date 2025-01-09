"""
Test edge cases
"""

import logging

from pydantic import ValidationError
from pytest import skip

from poprox_concepts.api.recommendations import RecommendationRequest
from poprox_recommender.config import allow_data_test_failures
from poprox_recommender.paths import project_root

logger = logging.getLogger(__name__)


def test_no_clicks_case():
    """
    Test the case where the user has no click history.
    """
    test_dir = project_root() / "tests"
    req_f = test_dir / "request_data" / "request_body.json"
    if allow_data_test_failures() and not req_f.exists():
        skip("Request file does not exist")

    try:
        req = RecommendationRequest.model_validate_json(req_f.read_text())
        req.interest_profile.click_history = []

        assert isinstance(req.interest_profile.click_history, list)
        assert len(req.interest_profile.click_history) == 0

        print("No clicks case passed")
    except ValidationError as e:
        raise e

    logger.info("No clicks case passed validation.")


def test_no_onboarding_case():
    """
    Test the case where the user has no onboarding topics.
    """
    test_dir = project_root() / "tests"
    req_f = test_dir / "request_data" / "request_body.json"
    if allow_data_test_failures() and not req_f.exists():
        skip("Request file does not exist")

    try:
        req = RecommendationRequest.model_validate_json(req_f.read_text())
        req.interest_profile.onboarding_topics = []

        assert isinstance(req.interest_profile.onboarding_topics, list)
        assert len(req.interest_profile.onboarding_topics) == 0

        print("No onboarding case passed")
    except ValidationError as e:
        raise e

    logger.info("No onboarding topics case passed validation.")
