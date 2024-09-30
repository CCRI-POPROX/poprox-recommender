"""
Test the topic calibration logic.
"""

import logging

from pytest import skip

from poprox_concepts import ArticleSet
from poprox_concepts.api.recommendations import RecommendationRequest
from poprox_recommender.paths import project_root
from poprox_recommender.recommenders import select_articles

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def test_request_with_topic_calibrator():
    test_dir = project_root() / "tests"
    req_f = test_dir / "request_data" / "medium_request.json"

    req = RecommendationRequest.model_validate_json(req_f.read_text())

    try:
        base_outputs = select_articles(
            ArticleSet(articles=req.todays_articles),
            ArticleSet(articles=req.past_articles),
            req.interest_profile,
        )
        topic_calibrated_outputs = select_articles(
            ArticleSet(articles=req.todays_articles),
            ArticleSet(articles=req.past_articles),
            req.interest_profile,
            pipeline_params={"pipeline": "topic-cali"},
        )
    except FileNotFoundError:
        # FIXME: when we have configuration files, separate "cannot load" from "cannot run"
        skip("model data not available")

    # do we get recommendations?
    tco_recs = topic_calibrated_outputs.default.articles
    bo_recs = base_outputs.default.articles
    assert len(tco_recs) > 0
    assert len(bo_recs) == len(tco_recs)

    base_article_ids = [article.article_id for article in bo_recs]
    calibrated_article_ids = [article.article_id for article in tco_recs]

    # are the recommendation lists different?
    assert base_article_ids != calibrated_article_ids
