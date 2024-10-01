"""
Test the topic calibration logic.
"""

import logging

from pytest import skip, xfail

from poprox_concepts import ArticleSet
from poprox_concepts.api.recommendations import RecommendationRequest
from poprox_recommender.config import allow_data_test_failures
from poprox_recommender.paths import project_root
from poprox_recommender.recommenders import PipelineLoadError, select_articles
from poprox_recommender.topics import user_locality_preference, user_topic_preference

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def test_request_with_topic_calibrator():
    test_dir = project_root() / "tests"
    req_f = test_dir / "request_data" / "request_body.json"
    if allow_data_test_failures() and not req_f.exists():
        skip("request file does not exist")

    req = RecommendationRequest.model_validate_json(req_f.read_text())
    req.interest_profile.click_topic_counts = user_topic_preference(
        req.past_articles, req.interest_profile.click_history
    )

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
    except PipelineLoadError as e:
        if allow_data_test_failures():
            xfail("data not pulled")
        else:
            raise e

    # do we get recommendations?
    tco_recs = topic_calibrated_outputs.default.articles
    bo_recs = base_outputs.default.articles
    assert len(tco_recs) > 0
    assert len(bo_recs) == len(tco_recs)

    base_article_ids = [article.article_id for article in bo_recs]
    calibrated_article_ids = [article.article_id for article in tco_recs]

    # are the recommendation lists different?
    assert base_article_ids != calibrated_article_ids


def test_request_with_locality_calibrator():
    test_dir = project_root() / "tests"
    req_f = test_dir / "request_data" / "request_body.json"
    if allow_data_test_failures() and not req_f.exists():
        skip("request file does not exist")

    req = RecommendationRequest.model_validate_json(req_f.read_text())
    req.interest_profile.click_locality_counts = user_locality_preference(
        req.past_articles, req.interest_profile.click_history
    )
    try:
        base_outputs = select_articles(
            ArticleSet(articles=req.todays_articles),
            ArticleSet(articles=req.past_articles),
            req.interest_profile,
        )
        locality_calibrated_outputs = select_articles(
            ArticleSet(articles=req.todays_articles),
            ArticleSet(articles=req.past_articles),
            req.interest_profile,
            pipeline_params={"pipeline": "locality-cali"},
        )
    except PipelineLoadError as e:
        if allow_data_test_failures():
            xfail("data not pulled")
        else:
            raise e

    # do we get recommendations?
    tco_recs = locality_calibrated_outputs.default.articles
    bo_recs = base_outputs.default.articles
    assert len(tco_recs) > 0
    assert len(bo_recs) == len(tco_recs)

    base_article_ids = [article.article_id for article in bo_recs]
    calibrated_article_ids = [article.article_id for article in tco_recs]

    # are the recommendation lists different?
    assert base_article_ids != calibrated_article_ids
