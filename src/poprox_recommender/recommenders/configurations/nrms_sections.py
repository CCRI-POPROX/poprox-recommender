from uuid import UUID

from lenskit.pipeline import PipelineBuilder

from poprox_concepts.domain import ArticlePackage, CandidateSet, InterestProfile
from poprox_recommender.components.embedders import NRMSArticleEmbedder
from poprox_recommender.components.filters.impression import ImpressionFilter
from poprox_recommender.components.embedders.article import NRMSArticleEmbedderConfig
from poprox_recommender.components.embedders.user import NRMSUserEmbedder, NRMSUserEmbedderConfig
from poprox_recommender.components.embedders.user_article_feedback import (
    UserArticleFeedbackConfig,
    UserArticleFeedbackEmbedder,
)
from poprox_recommender.components.embedders.user_topic_prefs import UserOnboardingConfig, UserOnboardingEmbedder
from poprox_recommender.components.joiners.score import ScoreFusion
from poprox_recommender.components.rankers.sectionizer import Sectionizer, SectionizerConfig
from poprox_recommender.components.scorers.article import ArticleScorer
from poprox_recommender.paths import model_file_path

TOP_NEWS_PACKAGE_ID = UUID("72bb7674-7bde-4f3e-a351-ccdeae888502")


def configure(builder: PipelineBuilder, num_slots: int, device: str):
    # standard practice is to put these calls in this order, to reuse logic
    # Define pipeline inputs
    i_candidates = builder.create_input("candidate", CandidateSet)
    i_clicked = builder.create_input("clicked", CandidateSet)
    i_profile = builder.create_input("profile", InterestProfile)
    i_packages = builder.create_input("packages", list[ArticlePackage])
    i_impressed_ids = builder.create_input("impressed_article_ids", list[UUID])

    # Filter out articles user has already received (prevents duplicates)
    f_candidates = builder.add_component(
        "impression-filter",
        ImpressionFilter,
        candidates=i_candidates,
        impressed_article_ids=i_impressed_ids,
    )

    # Embed candidate and clicked articles
    ae_config = NRMSArticleEmbedderConfig(
        model_path=model_file_path("nrms-mind/news_encoder.safetensors"),
        device=device,
    )
    e_candidates = builder.add_component("candidate-embedder", NRMSArticleEmbedder, ae_config, article_set=f_candidates)
    e_clicked = builder.add_component(
        "history-NRMSArticleEmbedder", NRMSArticleEmbedder, ae_config, article_set=i_clicked
    )

    # Embed the user (historical clicks)
    ue_config = NRMSUserEmbedderConfig(
        model_path=model_file_path("nrms-mind/user_encoder.safetensors"),
        device=device,
    )
    e_user = builder.add_component(
        "user-embedder",
        NRMSUserEmbedder,
        ue_config,
        candidate_articles=e_candidates,
        clicked_articles=e_clicked,
        interest_profile=i_profile,
    )

    # Embed the positive user topic preferences
    ue_pos_topic_config = UserOnboardingConfig(
        model_path=model_file_path("nrms-mind/user_encoder.safetensors"),
        device=device,
        embedding_source="static",
        topic_embedding="nrms",
        topic_pref_values=[4, 5],
    )
    e_topic_positive = builder.add_component(
        "user-pos-topic-embedder",
        UserOnboardingEmbedder,
        ue_pos_topic_config,
        candidate_articles=e_candidates,
        clicked_articles=e_clicked,
        interest_profile=i_profile,
    )

    # Embed the negative user topic preferences
    ue_neg_topic_config = UserOnboardingConfig(
        model_path=model_file_path("nrms-mind/user_encoder.safetensors"),
        device=device,
        embedding_source="static",
        topic_embedding="nrms",
        topic_pref_values=[1, 2],
    )
    e_topic_negative = builder.add_component(
        "user-neg-topic-embedder",
        UserOnboardingEmbedder,
        ue_neg_topic_config,
        candidate_articles=e_candidates,
        clicked_articles=e_clicked,
        interest_profile=i_profile,
    )

    # Embed the user positive feedback
    ue_pos_fb_config = UserArticleFeedbackConfig(
        model_path=model_file_path("nrms-mind/user_encoder.safetensors"),
        device=device,
        feedback_type=True,
    )
    e_feedback_positive = builder.add_component(
        "user-pos-fb-embedder",
        UserArticleFeedbackEmbedder,
        ue_pos_fb_config,
        candidate_articles=e_candidates,
        interacted_articles=e_clicked,
        interest_profile=i_profile,
    )

    # Embed the user negative feedback
    ue_neg_fb_config = UserArticleFeedbackConfig(
        model_path=model_file_path("nrms-mind/user_encoder.safetensors"),
        device=device,
        feedback_type=False,
    )
    e_feedback_negative = builder.add_component(
        "user-neg-fb-embedder",
        UserArticleFeedbackEmbedder,
        ue_neg_fb_config,
        candidate_articles=e_candidates,
        interacted_articles=e_clicked,
        interest_profile=i_profile,
    )

    # Score articles based on interaction history
    n_scorer = builder.add_component(
        "scorer",
        ArticleScorer,
        candidate_articles=e_candidates,
        interest_profile=e_user,
    )

    # Score articles based on topic preferences
    positive_topic_score = builder.add_component(
        "positive_topic_score",
        ArticleScorer,
        candidate_articles=builder.node("candidate-embedder"),
        interest_profile=e_topic_positive,
    )

    negative_topic_score = builder.add_component(
        "negative_topic_score",
        ArticleScorer,
        candidate_articles=builder.node("candidate-embedder"),
        interest_profile=e_topic_negative,
    )

    topic_fusion = builder.add_component(
        "topic_fusion",
        ScoreFusion,
        {"combiner": "sub"},
        candidates1=positive_topic_score,
        candidates2=negative_topic_score,
    )

    # Score articles based on feedback
    positive_feedback_score = builder.add_component(
        "positive_feedback_score",
        ArticleScorer,
        candidate_articles=builder.node("candidate-embedder"),
        interest_profile=e_feedback_positive,
    )

    negative_feedback_score = builder.add_component(
        "negative_feedback_score",
        ArticleScorer,
        candidate_articles=builder.node("candidate-embedder"),
        interest_profile=e_feedback_negative,
    )

    feedback_fusion = builder.add_component(
        "feedback_fusion",
        ScoreFusion,
        {"combiner": "sub"},
        candidates1=positive_feedback_score,
        candidates2=negative_feedback_score,
    )

    # Combine topic scoring and feedback -> all explicit data
    explicit_fusion = builder.add_component(
        "explicit_fusion",
        ScoreFusion,
        {"combiner": "avg"},
        candidates1=topic_fusion,
        candidates2=feedback_fusion,
    )

    # Combine click and explicit feedback -> all preference
    fusion = builder.add_component(
        "fusion",
        ScoreFusion,
        {"combiner": "avg", "weight1": 1, "weight2": 2},
        candidates1=n_scorer,
        candidates2=explicit_fusion,
    )

    # Sectionizer
    s_config = SectionizerConfig(
        max_top_news=3,
        max_topic_sections=3,
        max_articles_per_topic=3,
        max_misc_articles=3,
    )

    builder.add_component(
        "recommender",
        Sectionizer,
        s_config,
        candidate_set=fusion,
        article_packages=i_packages,
        interest_profile=i_profile,
    )
