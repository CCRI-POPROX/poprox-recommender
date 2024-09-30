import logging

from pytest import mark, skip

from poprox_concepts import ArticleSet
from poprox_concepts.api.recommendations import RecommendationRequest
from poprox_recommender.config import allow_data_test_failures
from poprox_recommender.paths import project_root
from poprox_recommender.recommenders import PipelineLoadError, select_articles

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@mark.xfail(condition=allow_data_test_failures(), raises=PipelineLoadError, reason="data not pulled")
def test_request_with_softmax_sampler():
    test_dir = project_root() / "tests"
    req_f = test_dir / "request_data" / "medium_request.json"

    req = RecommendationRequest.model_validate_json(req_f.read_text())

    base_outputs = select_articles(
        ArticleSet(articles=req.todays_articles),
        ArticleSet(articles=req.past_articles),
        req.interest_profile,
    )
    sampled_outputs = select_articles(
        ArticleSet(articles=req.todays_articles),
        ArticleSet(articles=req.past_articles),
        req.interest_profile,
        pipeline_params={"pipeline": "softmax"},
    )

    # do we get recommendations?
    softmax_recs = sampled_outputs.default.articles
    base_recs = base_outputs.default.articles
    assert len(softmax_recs) > 0
    assert len(base_recs) == len(softmax_recs)

    base_article_ids = [article.article_id for article in base_recs]
    sampled_article_ids = [article.article_id for article in softmax_recs]

    # are the recommendation lists different?
    assert base_article_ids != sampled_article_ids
