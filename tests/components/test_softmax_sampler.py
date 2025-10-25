import logging

from pytest import skip, xfail

from poprox_concepts.api.recommendations.v4 import RecommendationRequestV4
from poprox_recommender.config import allow_data_test_failures
from poprox_recommender.paths import project_root
from poprox_recommender.recommenders import PipelineLoadError, select_articles

logger = logging.getLogger(__name__)


def test_request_with_softmax_sampler():
    test_dir = project_root() / "tests"
    req_f = test_dir / "request_data" / "request_body.json"
    if allow_data_test_failures() and not req_f.exists():
        skip("request file does not exist")

    req = RecommendationRequestV4.model_validate_json(req_f.read_text())

    try:
        base_outputs = select_articles(
            req.candidates,
            req.interacted,
            req.interest_profile,
        )
        sampled_outputs = select_articles(
            req.candidates,
            req.interacted,
            req.interest_profile,
            pipeline_params={"pipeline": "softmax"},
        )
    except PipelineLoadError as e:
        if allow_data_test_failures():
            xfail("data not pulled")
        else:
            raise e

    # do we get recommendations?
    softmax_recs, _ = sampled_outputs
    base_recs, _ = base_outputs
    assert len(softmax_recs.impressions) > 0
    assert len(base_recs.impressions) == len(softmax_recs.impressions)

    base_article_ids = [impression.article.article_id for impression in base_recs.impressions]
    sampled_article_ids = [impression.article.article_id for impression in softmax_recs.impressions]

    # are the recommendation lists different?
    assert base_article_ids != sampled_article_ids
