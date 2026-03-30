from uuid import UUID

from lenskit.pipeline import PipelineBuilder

from poprox_concepts.domain import ArticlePackage, CandidateSet, InterestProfile
from poprox_recommender.components.embedders import NRMSArticleEmbedder
from poprox_recommender.components.embedders.article import NRMSArticleEmbedderConfig
from poprox_recommender.components.embedders.user import NRMSUserEmbedder, NRMSUserEmbedderConfig
from poprox_recommender.components.embedders.user_article_feedback import (
    UserArticleFeedbackConfig,
    UserArticleFeedbackEmbedder,
)
from poprox_recommender.components.embedders.user_topic_prefs import UserOnboardingConfig, UserOnboardingEmbedder
from poprox_recommender.components.filters.duplicate import DuplicateFilter
from poprox_recommender.components.filters.impression import ImpressionFilter
from poprox_recommender.components.filters.seeds import PreviousSectionsFilter
from poprox_recommender.components.filters.topic import TopicFilter
from poprox_recommender.components.joiners.fill import FillConfig, FillRecs
from poprox_recommender.components.joiners.score import ScoreFusion
from poprox_recommender.components.rankers.topk import TopkConfig, TopkRanker
from poprox_recommender.components.scorers.article import ArticleScorer
from poprox_recommender.components.sections.combine import AddSection, AddSectionConfig
from poprox_recommender.components.selectors.top_news import TopStoryCandidates
from poprox_recommender.components.selectors.topical import TopicalCandidates, TopicalCandidatesConfig
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

    # Sections
    yts_candidates = builder.add_component(
        "yts_candidates",
        TopStoryCandidates,
        candidate_articles=fusion,
        article_packages=i_packages,
    )

    yts_filtered = builder.add_component(
        "yts_filtered", TopicFilter, candidates=yts_candidates, interest_profile=i_profile
    )

    yts_topk_filtered = builder.add_component(
        "yts_topk_filtered", TopkRanker, TopkConfig(num_slots=3), candidate_articles=yts_filtered
    )

    # The maximum overlap with the articles chosen above is self.config.max_articles,
    # so here we pull twice as many to cover the worst case
    yts_topk_unfiltered = builder.add_component(
        "yts_topk_unfiltered", TopkRanker, TopkConfig(num_slots=6), candidate_articles=yts_candidates
    )

    yts_fill = builder.add_component(
        "yts_fill", FillRecs, FillConfig(num_slots=3), recs1=yts_topk_filtered, recs2=yts_topk_unfiltered
    )

    yts_config = AddSectionConfig(title="Your Top Stories", personalized=True)
    yts_sections = builder.add_component("top_stories", AddSection, yts_config, new_section=yts_fill)

    # Topical Sections
    topic1_deduped = builder.add_component("topic1_deduped", DuplicateFilter, candidate=fusion, sections=yts_sections)

    topic1_candidates = builder.add_component(
        "topic1_candidates",
        TopicalCandidates,
        TopicalCandidatesConfig(min_candidates=3),
        candidate_set=topic1_deduped,
        interest_profile=i_profile,
        sections=yts_sections,
    )
    topic1_topk = builder.add_component(
        "topic1_topk", TopkRanker, TopkConfig(num_slots=3), candidate_articles=topic1_candidates
    )

    topic1_sections = builder.add_component(
        "topic1_sections",
        AddSection,
        AddSectionConfig(personalized=True),
        new_section=topic1_topk,
        existing_sections=yts_sections,
    )

    topic2_deduped = builder.add_component(
        "topic2_deduped", DuplicateFilter, candidate=fusion, sections=topic1_sections
    )

    topic2_candidates = builder.add_component(
        "topic2_candidates",
        TopicalCandidates,
        TopicalCandidatesConfig(min_candidates=3),
        candidate_set=topic2_deduped,
        interest_profile=i_profile,
        sections=topic1_sections,
    )
    topic2_topk = builder.add_component(
        "topic2_topk", TopkRanker, TopkConfig(num_slots=3), candidate_articles=topic2_candidates
    )

    topic2_sections = builder.add_component(
        "topic2_sections",
        AddSection,
        AddSectionConfig(personalized=True),
        new_section=topic2_topk,
        existing_sections=topic1_sections,
    )

    topic3_deduped = builder.add_component(
        "topic3_deduped", DuplicateFilter, candidate=fusion, sections=topic2_sections
    )

    topic3_candidates = builder.add_component(
        "topic3_candidates",
        TopicalCandidates,
        TopicalCandidatesConfig(min_candidates=3),
        candidate_set=topic3_deduped,
        interest_profile=i_profile,
        sections=topic2_sections,
    )
    topic3_topk = builder.add_component(
        "topic3_topk", TopkRanker, TopkConfig(num_slots=3), candidate_articles=topic3_candidates
    )

    topic3_sections = builder.add_component(
        "topic3_sections",
        AddSection,
        AddSectionConfig(personalized=True),
        new_section=topic3_topk,
        existing_sections=topic2_sections,
    )

    # In Other News section
    ion_deduped = builder.add_component("ion_deduped", DuplicateFilter, candidate=fusion, sections=topic3_sections)

    ion_narrowed = builder.add_component(
        "ion_narrowed",
        PreviousSectionsFilter,
        candidate=ion_deduped,
        article_packages=i_packages,
        sections=topic3_sections,
    )

    ion_filtered = builder.add_component(
        "ion_topic_filtered", TopicFilter, candidates=ion_narrowed, interest_profile=i_profile
    )

    ion_topk_filtered = builder.add_component(
        "ion_topk_filtered", TopkRanker, TopkConfig(num_slots=3), candidate_articles=ion_filtered
    )

    # The maximum overlap with the articles chosen above is self.config.max_articles,
    # so here we pull twice as many to cover the worst case
    ion_topk_unfiltered = builder.add_component(
        "ion_topk_unfiltered", TopkRanker, TopkConfig(num_slots=6), candidate_articles=ion_narrowed
    )

    ion_fill = builder.add_component(
        "ion_fill", FillRecs, FillConfig(num_slots=3), recs1=ion_topk_filtered, recs2=ion_topk_unfiltered
    )

    ion_config = AddSectionConfig(title="In Other News", personalized=True)
    builder.add_component(
        "recommender", AddSection, ion_config, new_section=ion_fill, existing_sections=topic3_sections
    )
