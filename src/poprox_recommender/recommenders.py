# pyright: basic
import logging
from typing import Any

from lenskit.pipeline import Pipeline, PipelineState

from poprox_concepts import CandidateSet, InterestProfile
from poprox_recommender.components.diversifiers import (
    # LocalityCalibrator,
    MMRDiversifier,
    PFARDiversifier,
    TopicCalibrator,
)
from poprox_recommender.components.embedders import NRMSArticleEmbedder, NRMSUserEmbedder
from poprox_recommender.components.embedders.topic_wise_user import UserOnboardingEmbedder
from poprox_recommender.components.filters import TopicFilter
from poprox_recommender.components.joiners import Fill, ReciprocalRankFusion
from poprox_recommender.components.rankers.topk import TopkRanker
from poprox_recommender.components.samplers import SoftmaxSampler, UniformSampler
from poprox_recommender.components.scorers import ArticleScorer
from poprox_recommender.config import default_device
from poprox_recommender.paths import model_file_path

logger = logging.getLogger(__name__)


_cached_pipelines = None


class PipelineLoadError(Exception):
    """
    Exception raised when a pipeline cannot be loaded or instantiated, to
    separate those errors from errors running the pipeline.
    """


def select_articles(
    candidate_articles: CandidateSet,
    clicked_articles: CandidateSet,
    interest_profile: InterestProfile,
    pipeline_params: dict[str, Any] | None = None,
) -> PipelineState:
    """
    Select articles with default recommender configuration.  It returns a
    pipeline state whose ``default`` is the final list of recommendations.
    """
    available_pipelines = recommendation_pipelines(device=default_device())
    pipeline = available_pipelines["nrms"]

    if pipeline_params and "pipeline" in pipeline_params:
        pipeline_name = pipeline_params["pipeline"]
        pipeline = available_pipelines[pipeline_name]

    recs = pipeline.node("recommender")
    topk = pipeline.node("ranker", missing="none")
    if topk is None:
        wanted = (recs,)
    else:
        wanted = (topk, recs)

    return pipeline.run_all(*wanted, candidate=candidate_articles, clicked=clicked_articles, profile=interest_profile)


def recommendation_pipelines(device: str | None = None, num_slots: int = 10) -> dict[str, Pipeline]:
    global _cached_pipelines
    if device is None:
        device = default_device()
    logger.debug("loading pipeline components on device %s", device)

    if _cached_pipelines is None:
        try:
            _cached_pipelines = build_pipelines(num_slots=num_slots, device=device)
        except Exception as e:
            e.add_note("did you remember to `dvc pull`?")
            raise PipelineLoadError("could not instantiate pipelines", e)

    return _cached_pipelines


def build_pipelines(num_slots: int, device: str) -> dict[str, Pipeline]:
    """
    Create the default personalized recommendation pipeline.

    Args:
        num_slots: The number of items to recommend.
    """

    article_embedder = NRMSArticleEmbedder(model_file_path("nrms-mind/news_encoder.safetensors"), device)
    user_embedder = NRMSUserEmbedder(model_file_path("nrms-mind/user_encoder.safetensors"), device)
    topic_user_embedder_candidate = UserOnboardingEmbedder(
        model_file_path("nrms-mind/user_encoder.safetensors"),
        device,
        embedding_source="candidates",
        topic_embedding="avg",
    )
    topic_user_embedder_clicked = UserOnboardingEmbedder(
        model_file_path("nrms-mind/user_encoder.safetensors"),
        device,
        embedding_source="clicked",
        topic_embedding="avg",
    )
    topic_user_embedder_static = UserOnboardingEmbedder(
        model_file_path("nrms-mind/user_encoder.safetensors"),
        device,
        embedding_source="static",
        topic_embedding="avg",
    )
    topic_user_embedder_hybrid = UserOnboardingEmbedder(
        model_file_path("nrms-mind/user_encoder.safetensors"),
        device,
        embedding_source="hybrid",
        topic_embedding="avg",
    )

    topk_ranker = TopkRanker(num_slots=num_slots)
    mmr = MMRDiversifier(num_slots=num_slots)
    pfar = PFARDiversifier(num_slots=num_slots)
    # locality_calibrator = LocalityCalibrator(num_slots=num_slots)
    topic_calibrator = TopicCalibrator(num_slots=num_slots)
    sampler = SoftmaxSampler(num_slots=num_slots, temperature=30.0)

    nrms_pipe = build_pipeline(
        "plain-NRMS",
        article_embedder=article_embedder,
        user_embedder=user_embedder,
        ranker=topk_ranker,
        num_slots=num_slots,
    )

    nrms_onboarding_pipe_cadidate = build_pipeline(
        "plain-NRMS-with-onboarding-topics",
        article_embedder=article_embedder,
        user_embedder=topic_user_embedder_candidate,
        ranker=topk_ranker,
        num_slots=num_slots,
    )

    nrms_onboarding_pipe_clicked = build_pipeline(
        "plain-NRMS-with-onboarding-topics",
        article_embedder=article_embedder,
        user_embedder=topic_user_embedder_clicked,
        ranker=topk_ranker,
        num_slots=num_slots,
    )

    nrms_onboarding_pipe_static = build_pipeline(
        "plain-NRMS-with-onboarding-topics",
        article_embedder=article_embedder,
        user_embedder=topic_user_embedder_static,
        ranker=topk_ranker,
        num_slots=num_slots,
    )

    nrms_onboarding_pipe_hybrid = build_pipeline(
        "plain-NRMS-with-onboarding-topics",
        article_embedder=article_embedder,
        user_embedder=topic_user_embedder_hybrid,
        ranker=topk_ranker,
        num_slots=num_slots,
    )

    mmr_pipe = build_pipeline(
        "NRMS+MMR",
        article_embedder=article_embedder,
        user_embedder=user_embedder,
        ranker=mmr,
        num_slots=num_slots,
    )

    pfar_pipe = build_pipeline(
        "NRMS+PFAR",
        article_embedder=article_embedder,
        user_embedder=user_embedder,
        ranker=pfar,
        num_slots=num_slots,
    )

    topic_cali_pipe = build_pipeline(
        "NRMS+Topic+Calibration",
        article_embedder=article_embedder,
        user_embedder=user_embedder,
        ranker=topic_calibrator,
        num_slots=num_slots,
    )

    # locality_cali_pipe = build_pipeline(
    #     "NRMS+Locality+Calibration",
    #     article_embedder=article_embedder,
    #     user_embedder=user_embedder,
    #     ranker=locality_calibrator,
    #     num_slots=num_slots,
    # )

    softmax_pipe = build_pipeline(
        "NRMS+Softmax",
        article_embedder=article_embedder,
        user_embedder=user_embedder,
        ranker=sampler,
        num_slots=num_slots,
    )

    nrms_rrf_static_candidate = build_RRF_pipeline(
        "NRMS+RRF",
        article_embedder=article_embedder,
        user_embedder=topic_user_embedder_static,
        user_embedder2=topic_user_embedder_candidate,
        ranker=topk_ranker,
        num_slots=num_slots,
    )

    nrms_rrf_static_clicked = build_RRF_pipeline(
        "NRMS+RRF",
        article_embedder=article_embedder,
        user_embedder=topic_user_embedder_static,
        user_embedder2=topic_user_embedder_clicked,
        ranker=topk_ranker,
        num_slots=num_slots,
    )

    return {
        "nrms": nrms_pipe,
        "nrms-topics-candidate": nrms_onboarding_pipe_cadidate,
        "nrms-topics-clicked": nrms_onboarding_pipe_clicked,
        "nrms-topics-static": nrms_onboarding_pipe_static,
        "nrms-topics-hybrid": nrms_onboarding_pipe_hybrid,
        "mmr": mmr_pipe,
        "pfar": pfar_pipe,
        "topic-cali": topic_cali_pipe,
        # "locality-cali": locality_cali_pipe,
        "softmax": softmax_pipe,
        "nrms_rrf_static_candidate": nrms_rrf_static_candidate,
        "nrms_rrf_static_clicked": nrms_rrf_static_clicked,
    }


def build_pipeline(name, article_embedder, user_embedder, ranker, num_slots):
    article_scorer = ArticleScorer()
    topic_filter = TopicFilter()
    sampler = UniformSampler(num_slots=num_slots)
    fill = Fill(num_slots=num_slots)
    topk_ranker = TopkRanker(num_slots=num_slots)
    pipeline = Pipeline(name=name)

    # Define pipeline inputs
    candidates = pipeline.create_input("candidate", CandidateSet)
    clicked = pipeline.create_input("clicked", CandidateSet)
    profile = pipeline.create_input("profile", InterestProfile)

    # Compute embeddings
    e_cand = pipeline.add_component("candidate-embedder", article_embedder, article_set=candidates)
    e_click = pipeline.add_component("history-embedder", article_embedder, article_set=clicked)
    e_user = pipeline.add_component(
        "user-embedder",
        user_embedder,
        candidate_articles=candidates,
        clicked_articles=e_click,
        interest_profile=profile,
    )

    # Score and rank articles with diversification/calibration reranking
    o_scored = pipeline.add_component("scorer", article_scorer, candidate_articles=e_cand, interest_profile=e_user)
    o_topk = pipeline.add_component("ranker", topk_ranker, candidate_articles=o_scored, interest_profile=e_user)
    if ranker is topk_ranker:
        o_rank = o_topk
    else:
        o_rank = pipeline.add_component("reranker", ranker, candidate_articles=o_scored, interest_profile=e_user)

    # Fallback in case not enough articles came from the ranker
    o_filtered = pipeline.add_component("topic-filter", topic_filter, candidate=candidates, interest_profile=profile)
    o_sampled = pipeline.add_component("sampler", sampler, candidates1=o_filtered, candidates2=candidates)
    pipeline.add_component("recommender", fill, recs1=o_rank, recs2=o_sampled)

    return pipeline


def build_RRF_pipeline(name, article_embedder, user_embedder, user_embedder2, ranker, num_slots):
    article_scorer = ArticleScorer()
    rrf = ReciprocalRankFusion(num_slots=num_slots)
    topk_ranker = TopkRanker(num_slots=num_slots)

    pipeline = Pipeline(name=name)

    # Define pipeline inputs
    candidates = pipeline.create_input("candidate", CandidateSet)
    clicked = pipeline.create_input("clicked", CandidateSet)
    profile = pipeline.create_input("profile", InterestProfile)

    # Compute embeddings
    e_cand = pipeline.add_component("candidate-embedder", article_embedder, article_set=candidates)
    e_click = pipeline.add_component("history-embedder", article_embedder, article_set=clicked)

    # First user embedding strategy
    e_user_1 = pipeline.add_component(
        "user-embedder",
        user_embedder,
        candidate_articles=candidates,
        clicked_articles=e_click,
        interest_profile=profile,
    )

    # Score and rank articles with diversification/calibration reranking
    o_scored_1 = pipeline.add_component("scorer", article_scorer, candidate_articles=e_cand, interest_profile=e_user_1)
    o_topk_1 = pipeline.add_component("ranker", topk_ranker, candidate_articles=o_scored_1, interest_profile=e_user_1)
    if ranker is topk_ranker:
        o_rank_1 = o_topk_1
    else:
        o_rank_1 = pipeline.add_component("reranker", ranker, candidate_articles=o_scored_1, interest_profile=e_user_1)

    # Second user embedding strategy
    e_user_2 = pipeline.add_component(
        "user-embedder2",
        user_embedder2,
        candidate_articles=candidates,
        clicked_articles=e_click,
        interest_profile=profile,
    )

    o_scored_2 = pipeline.add_component("scorer2", article_scorer, candidate_articles=e_cand, interest_profile=e_user_2)
    o_topk_2 = pipeline.add_component("ranker2", topk_ranker, candidate_articles=o_scored_2, interest_profile=e_user_2)
    if ranker is topk_ranker:
        o_rank_2 = o_topk_2
    else:
        o_rank_2 = pipeline.add_component("reranker2", ranker, candidate_articles=o_scored_2, interest_profile=e_user_2)

    # Merge recommendations from each strategy
    pipeline.add_component("recommender", rrf, recs1=o_rank_1, recs2=o_rank_2)

    return pipeline
