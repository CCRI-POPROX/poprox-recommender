from lenskit.pipeline import PipelineBuilder

from poprox_concepts import CandidateSet, InterestProfile
from poprox_recommender.components.embedders import NRMSArticleEmbedder
from poprox_recommender.components.embedders.article import NRMSArticleEmbedderConfig
from poprox_recommender.components.embedders.user import NRMSUserEmbedder, NRMSUserEmbedderConfig
from poprox_recommender.components.embedders.user_article_feedback import (
    UserArticleFeedbackConfig,
    UserArticleFeedbackEmbedder,
)
from poprox_recommender.components.embedders.user_topic_prefs import (
    StaticDefinitionUserTopicEmbedder,
    UserTopicEmbedderConfig,
)
from poprox_recommender.components.filters.topic import TopicFilter
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

    # Filter articles based on topic preferences
    f_candidates = builder.add_component(
        "topic-filter", TopicFilter, candidates=i_candidates, interest_profile=i_profile
    )

    # Embed candidate and clicked articles
    ae_config = NRMSArticleEmbedderConfig(
        model_path=model_file_path("nrms-mind/news_encoder.safetensors"), device=device
    )
    e_candidates = builder.add_component("candidate-embedder", NRMSArticleEmbedder, ae_config, article_set=f_candidates)
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
    ue_config2 = UserTopicEmbedderConfig(
        model_path=model_file_path("nrms-mind/user_encoder.safetensors"),
        device=device,
        topic_embedding="nrms",
        topic_pref_values=[4, 5],
    )
    e_topic_positive = builder.add_component(
        "user-embedder2",
        StaticDefinitionUserTopicEmbedder,
        ue_config2,
        candidate_articles=e_candidates,
        clicked_articles=e_clicked,
        interest_profile=i_profile,
    )

    # Embed the user2 (topics)
    ue_config3 = UserTopicEmbedderConfig(
        model_path=model_file_path("nrms-mind/user_encoder.safetensors"),
        device=device,
        topic_embedding="nrms",
        topic_pref_values=[1, 2],
    )
    e_topic_negative = builder.add_component(
        "user-embedder3",
        StaticDefinitionUserTopicEmbedder,
        ue_config3,
        candidate_articles=e_candidates,
        clicked_articles=e_clicked,
        interest_profile=i_profile,
    )

    # Embed the user (feedbacks)
    ue_config4 = UserArticleFeedbackConfig(
        model_path=model_file_path("nrms-mind/user_encoder.safetensors"),
        device=device,
        feedback_type="positive",
    )
    e_feedback_positive = builder.add_component(
        "user-embedder4",
        UserArticleFeedbackEmbedder,
        ue_config4,
        candidate_articles=e_candidates,
        interacted_articles=e_clicked,
        interest_profile=i_profile,
    )

    # Embed the user2 (feedbacks)
    ue_config5 = UserArticleFeedbackConfig(
        model_path=model_file_path("nrms-mind/user_encoder.safetensors"),
        device=device,
        feedback_type="negative",
    )
    e_feedback_negative = builder.add_component(
        "user-embedder5",
        UserArticleFeedbackEmbedder,
        ue_config5,
        candidate_articles=e_candidates,
        interacted_articles=e_clicked,
        interest_profile=i_profile,
    )

    # Score and rank articles (history)
    n_scorer = builder.add_component("scorer", ArticleScorer, candidate_articles=e_candidates, interest_profile=e_user)

    # Score and rank articles (topics)
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

    # Score and rank articles (feedbacks)
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

    builder.add_component("recommender", TopkRanker, {"num_slots": num_slots}, candidate_articles=fusion)
