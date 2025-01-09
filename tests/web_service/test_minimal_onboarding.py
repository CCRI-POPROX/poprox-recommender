"""
Test minimal set of onboarding topics
"""

import logging

from pydantic import ValidationError
from pytest import skip

from poprox_concepts.api.recommendations import RecommendationRequest
from poprox_concepts.domain.account import AccountInterest
from poprox_recommender.config import allow_data_test_failures
from poprox_recommender.paths import project_root

logger = logging.getLogger(__name__)


def test_minimal_onboarding():
    test_dir = project_root() / "tests"
    req_f = test_dir / "request_data" / "request_body.json"
    if allow_data_test_failures() and not req_f.exists():
        skip("Request file does not exist")

    req = RecommendationRequest.model_validate_json(req_f.read_text())

    onboarding_topics = req.interest_profile.onboarding_topics

    try:
        if onboarding_topics:
            validated_topics = [AccountInterest.model_validate(topic) for topic in onboarding_topics]
            assert len(validated_topics) == len(onboarding_topics), "Validation for onboarding topics failed."
        else:
            assert onboarding_topics == []

    except ValidationError as e:
        raise e

    logger.info("Onboarding topics passed validation.")
