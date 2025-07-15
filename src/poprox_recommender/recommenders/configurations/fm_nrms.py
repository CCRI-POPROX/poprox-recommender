# pyright: basic

from lenskit.pipeline import PipelineBuilder

from poprox_concepts import CandidateSet, InterestProfile
from poprox_recommender.components.embedders import NRMSArticleEmbedder
from poprox_recommender.components.embedders.article import NRMSArticleEmbedderConfig
from poprox_recommender.components.embedders.fm_style_topic_pref import (
    UserExplicitTopicalEmbedder,
    UserExplicitTopicalEmbedderConfig,
    UserImplicitTopicalEmbedder,
    UserImplicitTopicalEmbedderConfig,
)
from poprox_recommender.components.embedders.topic_infused_article import (
    FMStyleArticleEmbedder,
)
from poprox_recommender.components.embedders.user import NRMSMultiVectorUserEmbedder, NRMSUserEmbedderConfig
from poprox_recommender.components.joiners.feature_combiner import FeatureCombiner
from poprox_recommender.components.rankers.topk import TopkRanker
from poprox_recommender.components.scorers.fm import FMScorer
from poprox_recommender.paths import model_file_path


def configure(builder: PipelineBuilder, num_slots: int, device: str):
    # Define pipeline inputs
    i_candidates = builder.create_input("candidate", CandidateSet)
    i_clicked = builder.create_input("clicked", CandidateSet)
    i_profile = builder.create_input("profile", InterestProfile)

    # Embed candidate and clicked articles
    ae_config = NRMSArticleEmbedderConfig(
        model_path=model_file_path("nrms-mind/news_encoder.safetensors"), device=device
    )
    e_candidates = builder.add_component(
        "candidate-embedder", FMStyleArticleEmbedder, ae_config, article_set=i_candidates
    )
    e_clicked = builder.add_component(
        "history-NRMSArticleEmbedder", NRMSArticleEmbedder, ae_config, article_set=i_clicked
    )

    # Embed the user
    ue_config = NRMSUserEmbedderConfig(model_path=model_file_path("nrms-mind/user_encoder.safetensors"), device=device)
    e_user_history = builder.add_component(
        "user-embedder",
        NRMSMultiVectorUserEmbedder,
        ue_config,
        candidate_articles=e_candidates,
        interacted_articles=e_clicked,
        interest_profile=i_profile,
    )

    # Embed the user (explicit topics)
    ue_config2 = UserExplicitTopicalEmbedderConfig(
        model_path=model_file_path("nrms-mind/user_encoder.safetensors"),
        device=device,
    )
    e_user_topic_explicit = builder.add_component(
        "user-embedder2",
        UserExplicitTopicalEmbedder,
        ue_config2,
        candidate_articles=e_candidates,
        interacted_articles=e_clicked,
        interest_profile=i_profile,
    )

    # Embed the user (implicit topics)
    ue_config3 = UserImplicitTopicalEmbedderConfig(
        model_path=model_file_path("nrms-mind/user_encoder.safetensors"),
        device=device,
    )
    e_user_topic_implicit = builder.add_component(
        "user-embedder3",
        UserImplicitTopicalEmbedder,
        ue_config3,
        candidate_articles=e_candidates,
        interacted_articles=e_clicked,
        interest_profile=i_profile,
    )

    # combined user
    e_user_combined_interest_profile = builder.add_component(
        "user-combined-embedder",
        FeatureCombiner,
        profiles_1=e_user_history,
        profiles_2=e_user_topic_explicit,
        profiles_3=e_user_topic_implicit,
    )

    # Score and rank articles
    n_scorer = builder.add_component(
        "scorer", FMScorer, candidate_articles=e_candidates, interest_profile=e_user_combined_interest_profile
    )
    builder.add_component("recommender", TopkRanker, {"num_slots": num_slots}, candidate_articles=n_scorer)
