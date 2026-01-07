"""
Validate edge case: user has no on boarding topic
"""

import logging
import warnings

from pydantic import ValidationError
from pytest import mark, skip

from poprox_concepts.api.recommendations import RecommendationRequest, RecommendationResponse
from poprox_recommender.config import allow_data_test_failures
from poprox_recommender.recommenders import discover_pipelines
from poprox_recommender.testing import RequestGenerator, mind_data
from poprox_recommender.testing import auto_service as service

logger = logging.getLogger(__name__)


@mark.parametrize("pipeline", discover_pipelines())
def test_no_onboarding(service, mind_data, pipeline):  # noqa: F811
    """
    Initialize request data
    """
    request_generator = RequestGenerator(mind_data)
    request_generator.add_candidates(100)
    request_generator.add_clicks(num_clicks=37, num_days=7)
    request_generator.add_topics([])
    request_generator.set_num_recs(10)
    req_body = request_generator.get_request()

    logger.info("sending request")
    response = service.request(req_body, pipeline)
    logger.info("response: %s", response.model_dump_json(indent=2))
    # do we have recommendations?
    recs = response.recommendations.impressions
    assert len(recs) > 0
    # do we have the correct number of recommendations
    assert len(recs) == request_generator.num_recs
    # are all recommendations unique?
    article_ids = [impression.article.article_id for impression in recs]
    assert len(article_ids) == len(set(article_ids))
