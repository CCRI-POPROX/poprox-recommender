"""
Test the topic calibration logic.
"""

import logging
from pathlib import Path

from poprox_concepts import ArticleSet
from poprox_concepts.api.recommendations import RecommendationRequest
from poprox_recommender.default import select_articles

logger = logging.getLogger(__name__)


def test_request_with_topic_calibrator():
    test_dir = Path(__file__)
    req_f = test_dir.parent / "medium_request.json"
    req = RecommendationRequest.model_validate_json(req_f.read_text())

    logger.info("generating recommendations")
    base_recs = select_articles(
        ArticleSet(articles=req.todays_articles),
        ArticleSet(articles=req.past_articles),
        req.interest_profile,
        req.num_recs,
    )
    topic_calibrated_recs = select_articles(
        ArticleSet(articles=req.todays_articles),
        ArticleSet(articles=req.past_articles),
        req.interest_profile,
        req.num_recs,
        algo_params={"diversity_algo": "topic-cali"},
    )

    # do we get recommendations?
    assert len(topic_calibrated_recs.articles) > 0

    assert len(base_recs.articles) == len(topic_calibrated_recs.articles)

    # check the topic calibrated recs are different from the base ones
    at_least_one_diff = False
    for i in range(len(base_recs.articles)):
        if base_recs.articles[i].article_id != topic_calibrated_recs.articles[i].article_id:
            at_least_one_diff = True
            break
    assert at_least_one_diff
