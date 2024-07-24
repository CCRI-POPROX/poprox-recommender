"""
Test the topic calibration logic.
"""

import logging
from pathlib import Path

from poprox_concepts import ArticleSet
from poprox_concepts.api.recommendations import RecommendationRequest
from poprox_recommender.default import select_articles

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def test_request_with_topic_calibrator():
    test_dir = Path(__file__)
    req_f = test_dir.parent / "medium_request.json"
    req = RecommendationRequest.model_validate_json(req_f.read_text())

    base_outputs = select_articles(
        ArticleSet(articles=req.todays_articles),
        ArticleSet(articles=req.past_articles),
        req.interest_profile,
        req.num_recs,
    )
    topic_calibrated_outputs = select_articles(
        ArticleSet(articles=req.todays_articles),
        ArticleSet(articles=req.past_articles),
        req.interest_profile,
        req.num_recs,
        algo_params={"diversity_algo": "topic-cali"},
    )

    # do we get recommendations?
    assert len(topic_calibrated_outputs.recs) > 0
    assert len(base_outputs.recs) == len(topic_calibrated_outputs.recs)

    base_article_ids = [article.article_id for article in base_outputs.recs]
    calibrated_article_ids = [article.article_id for article in topic_calibrated_outputs.recs]

    # are the recommendation lists different?
    assert base_article_ids != calibrated_article_ids
