"""
Test the POPROX API through direct call.
"""

import logging

from poprox_concepts import ArticleSet, ClickHistory
from poprox_concepts.api.recommendations import RecommendationRequest
from poprox_recommender.default import select_articles
from poprox_recommender.paths import project_root

logger = logging.getLogger(__name__)


def test_direct_basic_request():
    test_dir = project_root() / "tests"
    req_f = test_dir / "request_data" / "basic-request.json"
    req = RecommendationRequest.model_validate_json(req_f.read_text())

    logger.info("generating recommendations")
    outputs = select_articles(
        ArticleSet(articles=req.todays_articles),
        ArticleSet(articles=req.past_articles),
        req.interest_profile,
        req.num_recs,
    )
    # do we get recommendations?
    assert len(outputs.default.articles) > 0


def test_direct_basic_request_without_clicks():
    test_dir = project_root() / "tests"
    req_f = test_dir / "request_data" / "basic-request.json"
    req = RecommendationRequest.model_validate_json(req_f.read_text())

    logger.info("generating recommendations")

    profile = req.interest_profile
    profile.click_history = ClickHistory(article_ids=[])
    outputs = select_articles(
        ArticleSet(articles=req.todays_articles),
        ArticleSet(articles=req.past_articles),
        req.interest_profile,
        req.num_recs,
    )
    # do we get recommendations?
    assert len(outputs.default.articles) > 0
