"""
Validate edge case: user has no click history
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
def test_no_clicks(service, mind_data, pipeline):  # noqa: F811
    """
    Initialize request data
    """
    request_generator = RequestGenerator(mind_data)
    request_generator.add_candidates(150)
    request_generator.add_clicks(num_clicks=0, num_days=10)
    request_generator.add_topics(
        [
            "U.S. News",
            "World News",
            "General News",
            "Business",
            "Entertainment",
            "Sports",
            "Health",
            "Science",
            "Technology",
            "Oddities",
        ]
    )
    num_recs = 15
    request_generator.set_num_recs(num_recs)

    req_body = request_generator.get_request()

    logger.info("sending request")
    response = service.request(req_body, pipeline)
    logger.info("response: %s", response.model_dump_json(indent=2))
    # do we have recommendations?
    # get all recommendations from the sections
    recs = [imp for section in response.recommendations for imp in section.impressions]
    assert len(recs) > 0
    # do we have the correct number of recommendations
    assert len(recs) == request_generator.num_recs
    # are all recommendations unique?
    article_ids = [impression.article.article_id for impression in recs]
    assert len(article_ids) == len(set(article_ids))
