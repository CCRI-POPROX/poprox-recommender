# pyright: basic

from uuid import UUID

import numpy as np
from lenskit.pipeline import PipelineBuilder

from poprox_concepts import CandidateSet, InterestProfile
from poprox_recommender.components.embedders import NRMSArticleEmbedder
from poprox_recommender.components.embedders.article import NRMSArticleEmbedderConfig
from poprox_recommender.components.embedders.user import NRMSUserEmbedder, NRMSUserEmbedderConfig
from poprox_recommender.components.embedders.user_topic_prefs import UserOnboardingConfig, UserOnboardingEmbedder
from poprox_recommender.components.filters.image_selector_clip import GenericImageSelector
from poprox_recommender.components.joiners.score import ScoreFusion
from poprox_recommender.components.rankers.topk import TopkRanker
from poprox_recommender.components.scorers.article import ArticleScorer
from poprox_recommender.paths import model_file_path


def configure(builder: PipelineBuilder, num_slots: int, device: str):
    # standard practice is to put these calls in this order, to reuse logic
    # Define pipeline inputs
    i_candidates = builder.create_input("candidate", CandidateSet)
    i_clicked = builder.create_input("clicked", CandidateSet)
    i_profile = builder.create_input("profile", InterestProfile)
    # Pre-computed CLIP embeddings from V3 API
    embedding_lookup_table = builder.create_input("embedding_lookup", dict[UUID, dict[str, np.ndarray]])

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

    # Get the ranked recommendations (same as nrms_topic_scores)
    ranked_recommendations = builder.add_component(
        "ranked-recommendations", TopkRanker, {"num_slots": num_slots}, candidate_articles=fusion
    )

    # Add image personalization on top of the existing recommendations
    # This component selects personalized images for each article using CLIP embeddings
    builder.add_component(
        "recommender",
        GenericImageSelector,
        recommendations=ranked_recommendations,
        interest_profile=e_user,
        interacted_articles=i_clicked,
        embedding_lookup=embedding_lookup_table,
    )
