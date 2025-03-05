"""
Validate edge case: a user with only one click over several days
"""

import logging
import warnings

from pydantic import ValidationError
from pytest import mark, skip

from poprox_concepts.api.recommendations import RecommendationRequestV2, RecommendationResponseV2
from poprox_recommender.config import allow_data_test_failures
from poprox_recommender.recommenders import recommendation_pipelines
from poprox_recommender.testing import RequestGenerator, mind_data
from poprox_recommender.testing import auto_service as service

logger = logging.getLogger(__name__)
try:
    PIPELINES = recommendation_pipelines().keys()
except Exception as e:
    warnings.warn("failed to load models, did you run `dvc pull`?")
    if allow_data_test_failures():
        skip("recommendation pipelines unavailable", allow_module_level=True)
    else:
        raise e


@mark.parametrize("pipeline", PIPELINES)
def test_sparse_interaction_history(service, mind_data, pipeline):  # noqa: F811
    """
    Initialize request data
    """
    request_generator = RequestGenerator(mind_data)
    request_generator.add_candidates(100)
    request_generator.add_clicks(num_clicks=1, num_days=10)
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
    recs = response.recommendations.articles
    assert len(recs) > 0
    # do we have the correct number of recommendations
    assert len(recs) == request_generator.num_recs
    # are all recommendations unique?
    article_ids = [article.article_id for article in recs]
    assert len(article_ids) == len(set(article_ids))
