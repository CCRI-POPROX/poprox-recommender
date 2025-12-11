# pyright: basic

from uuid import UUID

from lenskit.pipeline import PipelineBuilder

from poprox_concepts.domain import ArticlePackage, CandidateSet, InterestProfile
from poprox_recommender.components.embedders import NRMSArticleEmbedder
from poprox_recommender.components.embedders.article import NRMSArticleEmbedderConfig
from poprox_recommender.components.embedders.user import NRMSUserEmbedder, NRMSUserEmbedderConfig
from poprox_recommender.components.embedders.user_topic_prefs import UserOnboardingConfig, UserOnboardingEmbedder
from poprox_recommender.components.filters.package import PackageFilter, PackageFilterConfig
from poprox_recommender.components.joiners.score import ScoreFusion
from poprox_recommender.components.rankers.top_news_placer import TopNewsPlacer, TopNewsPlacerConfig
from poprox_recommender.components.rankers.topk import TopkRanker
from poprox_recommender.components.scorers.article import ArticleScorer
from poprox_recommender.paths import model_file_path

# Top News Package ID (from platform)
TOP_NEWS_PACKAGE_ID = UUID("72bb7674-7bde-4f3e-a351-ccdeae888502")


def configure(builder: PipelineBuilder, num_slots: int, device: str):
    # standard practice is to put these calls in this order, to reuse logic
    # Define pipeline inputs
    i_candidates = builder.create_input("candidate", CandidateSet)
    i_clicked = builder.create_input("clicked", CandidateSet)
    i_profile = builder.create_input("profile", InterestProfile)
    i_packages = builder.create_input("packages", list[ArticlePackage])

    # Extract Top News Articles from Packages
    n_top_news_candidates = builder.add_component(
        "top-news-filter",
        PackageFilter,
        PackageFilterConfig(package_entity_id=TOP_NEWS_PACKAGE_ID),
        candidate_articles=i_candidates,
        article_packages=i_packages,
    )

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

    # Rank top candidates based on fused scores
    n_ranker = builder.add_component("ranker", TopkRanker, {"num_slots": num_slots}, candidate_articles=fusion)

    # Place top news at beginning (slots 1-3), followed by personalized recommendations (slots 4-12)
    placer_config = TopNewsPlacerConfig(
        max_top_news=3,
        total_slots=num_slots,
    )
    builder.add_component(
        "recommender",
        TopNewsPlacer,
        placer_config,
        ranked_articles=n_ranker,
        top_news_candidates=n_top_news_candidates,
    )
