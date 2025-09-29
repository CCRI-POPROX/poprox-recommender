# pyright: basic

import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from lenskit.pipeline import Component, PipelineBuilder

from poprox_concepts import CandidateSet, InterestProfile
from poprox_concepts.domain import RecommendationList
from poprox_recommender.components.embedders import NRMSArticleEmbedder
from poprox_recommender.components.embedders.article import NRMSArticleEmbedderConfig
from poprox_recommender.components.embedders.user import NRMSUserEmbedder, NRMSUserEmbedderConfig
from poprox_recommender.components.embedders.user_topic_prefs import UserOnboardingConfig, UserOnboardingEmbedder
from poprox_recommender.components.joiners.score import ScoreFusion
from poprox_recommender.components.rankers.openai_ranker import LLMRanker, LLMRankerConfig
from poprox_recommender.components.rankers.topk import TopkRanker
from poprox_recommender.components.rewriters.openai_rewriter import LLMRewriter, LLMRewriterConfig
from poprox_recommender.components.scorers.article import ArticleScorer
from poprox_recommender.paths import model_file_path


class NRMSWithUserModel(Component):
    """Component that combines NRMS baseline ranking with LLMRanker user model generation."""

    def __call__(
        self,
        nrms_ranking: RecommendationList,
        candidate_articles: CandidateSet,
        interest_profile: InterestProfile,
        articles_clicked: Optional[CandidateSet] = None,
    ) -> tuple[RecommendationList, str, str, dict, dict]:
        component_meta: Dict[str, Any] = {
            "component": "ranker",
            "implementation": "nrms_with_user_model",
            "start_time": datetime.now(timezone.utc).isoformat(),
            "status": "in_progress",
        }
        component_start = time.perf_counter()

        def finalize_component(status: str, exc: Exception | None = None) -> Dict[str, Any]:
            component_meta["status"] = status
            if exc is not None:
                component_meta["error_type"] = type(exc).__name__
                component_meta["error_message"] = str(exc)
            component_meta["end_time"] = datetime.now(timezone.utc).isoformat()
            component_meta["duration_seconds"] = time.perf_counter() - component_start
            if status == "success":
                component_meta["error_count"] = 0
            elif status == "error":
                component_meta.setdefault("error_count", 1)
            return component_meta.copy()

        try:
            # Create an LLMRanker instance just to generate the user model
            llm_ranker = LLMRanker(LLMRankerConfig())

            # Generate user model using LLMRanker's method
            profile_str = llm_ranker._structure_interest_profile(interest_profile, articles_clicked)
            user_model = llm_ranker._build_user_model(profile_str)
            request_id = str(interest_profile.profile_id)

            component_meta["num_candidates"] = len(candidate_articles.articles)
            component_meta["num_recommendations"] = len(nrms_ranking.articles)

            # Get the metrics from the LLMRanker instance (from user model generation)
            ranker_metrics = llm_ranker.llm_metrics

            component_snapshot = finalize_component("success")

            # Return the NRMS ranking with the LLM-generated user model and metrics
            return (nrms_ranking, user_model, request_id, ranker_metrics, {"ranker": component_snapshot})
        except Exception as exc:  # pragma: no cover - defensive logging for production observability
            finalize_component("error", exc)
            raise

##TODO:
# allow weigths for the scores (1/-1)


def configure(builder: PipelineBuilder, num_slots: int, device: str):
    # standard practice is to put these calls in this order, to reuse logic
    # Define pipeline inputs
    i_candidates = builder.create_input("candidate", CandidateSet)
    i_clicked = builder.create_input("clicked", CandidateSet)
    i_profile = builder.create_input("profile", InterestProfile)

    # Embed candidate and clicked articles
    ae_config = NRMSArticleEmbedderConfig(
        model_path=model_file_path("nrms-mind/news_encoder.safetensors"), device=device
    )
    e_candidates = builder.add_component("candidate-embedder", NRMSArticleEmbedder, ae_config, article_set=i_candidates)
    e_clicked = builder.add_component(
        "history-NRMSArticleEmbedder", NRMSArticleEmbedder, ae_config, article_set=i_clicked
    )

    # Embed the user (historical clicks)
    ue_config = NRMSUserEmbedderConfig(model_path=model_file_path("nrms-mind/user_encoder.safetensors"), device=device)
    e_user = builder.add_component(
        "user-embedder",
        NRMSUserEmbedder,
        ue_config,
        candidate_articles=e_candidates,
        clicked_articles=e_clicked,
        interest_profile=i_profile,
    )

    # Embed the user (topics)
    ue_config2 = UserOnboardingConfig(
        model_path=model_file_path("nrms-mind/user_encoder.safetensors"),
        device=device,
        embedding_source="static",
        topic_embedding="nrms",
        topic_pref_values=[4, 5],
    )
    e_user_positive = builder.add_component(
        "pos-topic-embedder",
        UserOnboardingEmbedder,
        ue_config2,
        candidate_articles=e_candidates,
        clicked_articles=e_clicked,
        interest_profile=i_profile,
    )

    # Embed the user2 (topics)
    ue_config3 = UserOnboardingConfig(
        model_path=model_file_path("nrms-mind/user_encoder.safetensors"),
        device=device,
        embedding_source="static",
        topic_embedding="nrms",
        topic_pref_values=[1, 2],
    )
    e_user_negative = builder.add_component(
        "neg-topic-embedder",
        UserOnboardingEmbedder,
        ue_config3,
        candidate_articles=e_candidates,
        clicked_articles=e_clicked,
        interest_profile=i_profile,
    )

    # Score and rank articles (history)
    n_scorer = builder.add_component("scorer", ArticleScorer, candidate_articles=e_candidates, interest_profile=e_user)

    # Score and rank articles (topics)
    positive_topic_score = builder.add_component(
        "positive_topic_score",
        ArticleScorer,
        candidate_articles=builder.node("candidate-embedder"),
        interest_profile=e_user_positive,
    )

    negative_topic_score = builder.add_component(
        "negative_topic_score",
        ArticleScorer,
        candidate_articles=builder.node("candidate-embedder"),
        interest_profile=e_user_negative,
    )

    topic_fusion = builder.add_component(
        "topic_fusion",
        ScoreFusion,
        {"combiner": "sub"},
        candidates1=positive_topic_score,
        candidates2=negative_topic_score,
    )

    # Combine click and topic scoring
    fusion = builder.add_component(
        "fusion", ScoreFusion, {"combiner": "avg"}, candidates1=n_scorer, candidates2=topic_fusion
    )

    # NRMS baseline ranking
    nrms_ranking = builder.add_component("nrms-ranker", TopkRanker, {"num_slots": num_slots}, candidate_articles=fusion)

    # Combine NRMS ranking with LLM user model
    ranker_output = builder.add_component(
        "ranker-with-user-model",
        NRMSWithUserModel,
        nrms_ranking=nrms_ranking,
        candidate_articles=i_candidates,
        interest_profile=i_profile,
        articles_clicked=i_clicked,
    )

    # LLM-based rewriting
    rewrite_cfg = LLMRewriterConfig()
    builder.add_component(
        "recommender",
        LLMRewriter,
        rewrite_cfg,
        ranker_output=ranker_output,
    )
