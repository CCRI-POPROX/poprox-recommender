"""
Test basic request by initializing test data.
"""

import logging

from pytest import mark

from poprox_recommender.recommenders import discover_pipelines
from poprox_recommender.testing import RequestGenerator, mind_data
from poprox_recommender.testing import auto_service as service

logger = logging.getLogger(__name__)


@mark.parametrize("pipeline", discover_pipelines())
def test_basic_request(service, mind_data, pipeline):  # noqa: F811
    """
    Initialize request data
    """
    request_generator = RequestGenerator(mind_data)
    request_generator.add_candidates(100)
    request_generator.add_clicks(num_clicks=37, num_days=7)
    request_generator.add_topics(
        [
            "Science",
            "Technology",
            "Sports",
            "Lifestyle",
            "Oddities",
        ]
    )
    request_generator.set_num_recs(10)
    req_body = request_generator.get_request()

    logger.info("sending request")
    response = service.request(req_body, pipeline)
    logger.info("response: %s", response.model_dump_json(indent=2))
    # do we have recommendations?
    sections = response.recommendations
    recs = sections[0].recommendations.impressions
    assert len(recs) > 0
    # do we have the correct number of recommendations
    assert len(recs) == request_generator.num_recs
    # are all recommendations unique?
    article_ids = [impression.article.article_id for impression in recs]
    assert len(article_ids) == len(set(article_ids))
