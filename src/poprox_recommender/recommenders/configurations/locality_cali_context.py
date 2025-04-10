# pyright: basic

from lenskit.pipeline import PipelineBuilder

from poprox_concepts import CandidateSet, InterestProfile
from poprox_recommender.components.diversifiers import (
    LocalityCalibrator,
)
from poprox_recommender.components.embedders import (
    NRMSArticleEmbedder,
    NRMSUserEmbedder,
)
from poprox_recommender.components.embedders.article import NRMSArticleEmbedderConfig
from poprox_recommender.components.embedders.topic_wise_user import UserOnboardingConfig, UserOnboardingEmbedder
from poprox_recommender.components.embedders.user import NRMSUserEmbedderConfig
from poprox_recommender.components.filters.topic import TopicFilter
from poprox_recommender.components.generators.context import ContextGenerator
from poprox_recommender.components.joiners.fill import FillRecs
from poprox_recommender.components.joiners.score import ScoreFusion
from poprox_recommender.components.rankers.topk import TopkRanker
from poprox_recommender.components.samplers.uniform import UniformSampler
from poprox_recommender.components.scorers import ArticleScorer
from poprox_recommender.paths import model_file_path


def configure(builder: PipelineBuilder, num_slots: int, device: str):
    # standard practice is to put these calls in this order, to reuse logic
    # Define pipeline inputs
    i_candidates = builder.create_input("candidate", CandidateSet)
    i_clicked = builder.create_input("clicked", CandidateSet)
    i_profile = builder.create_input("profile", InterestProfile)

    # locality-calibration specific inputs
    theta_topic = builder.create_input("theta_topic", float, None)
    theta_locality = builder.create_input("theta_locality", float, None)

    # Embed candidate and clicked articles
    ae_config = NRMSArticleEmbedderConfig(
        model_path=model_file_path("nrms-mind/news_encoder.safetensors"), device=device
    )
    e_candidates = builder.add_component("candidate-embedder", NRMSArticleEmbedder, ae_config, article_set=i_candidates)
    e_clicked = builder.add_component(
        "history-NRMSArticleEmbedder", NRMSArticleEmbedder, ae_config, article_set=i_clicked
    )

    # Embed the user
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
    )
    e_user2 = builder.add_component(
        "user-embedder2",
        UserOnboardingEmbedder,
        ue_config2,
        candidate_articles=e_candidates,
        clicked_articles=e_clicked,
        interest_profile=i_profile,
    )

    # Score and rank articles# Score and rank articles (history)
    n_scorer = builder.add_component("scorer", ArticleScorer, candidate_articles=e_candidates, interest_profile=e_user)

    # Score and rank articles (topics)
    n_scorer2 = builder.add_component(
        "scorer2", ArticleScorer, candidate_articles=builder.node("candidate-embedder"), interest_profile=e_user2
    )
    # Combine click and topic scoring
    fusion = builder.add_component(
        "fusion", ScoreFusion, {"combiner": "avg"}, candidates1=n_scorer, candidates2=n_scorer2
    )

    _n_topk = builder.add_component("ranker", TopkRanker, {"num_slots": num_slots}, candidate_articles=fusion)
    n_reranker = builder.add_component(
        "reranker",
        LocalityCalibrator,
        {"num_slots": num_slots},
        candidate_articles=fusion,
        interest_profile=e_user,
        theta_topic=theta_topic,
        theta_locality=theta_locality,
    )
    n_context = builder.add_component(
        "generator",
        ContextGenerator,
        {},
        clicked=e_clicked,
        selected=n_reranker,
        interest_profile=i_profile,
    )

    # Fallback: sample from user topic interests
    n_topic_filter = builder.add_component(
        "topic-filter", TopicFilter, candidate=i_candidates, interest_profile=i_profile
    )
    n_sampler = builder.add_component("sampler", UniformSampler, candidates1=n_topic_filter, candidates2=i_candidates)

    # Combinei primary ranker and fallback
    builder.add_component("recommender", FillRecs, {"num_slots": num_slots}, recs1=n_context, recs2=n_sampler)
