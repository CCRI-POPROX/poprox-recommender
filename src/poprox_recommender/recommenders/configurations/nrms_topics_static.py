# pyright: basic

from lenskit.pipeline import PipelineBuilder

from poprox_concepts import CandidateSet, InterestProfile
from poprox_recommender.components.embedders import NRMSArticleEmbedder
from poprox_recommender.components.embedders.article import NRMSArticleEmbedderConfig
from poprox_recommender.components.embedders.user_topic_prefs import UserOnboardingConfig, UserOnboardingEmbedder
from poprox_recommender.components.filters.topic import TopicFilter
from poprox_recommender.components.joiners.fill import FillRecs
from poprox_recommender.components.rankers.topk import TopkRanker
from poprox_recommender.components.samplers.uniform import UniformSampler
from poprox_recommender.components.scorers.article import ArticleScorer
from poprox_recommender.paths import model_file_path


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

    # Embed the user
    ue_config = UserOnboardingConfig(
        model_path=model_file_path("nrms-mind/user_encoder.safetensors"),
        device=device,
        embedding_source="static",
        topic_embedding="nrms",
    )
    e_user = builder.add_component(
        "user-embedder",
        UserOnboardingEmbedder,
        ue_config,
        candidate_articles=e_candidates,
        clicked_articles=e_clicked,
        interest_profile=i_profile,
    )

    # Score and rank articles
    n_scorer = builder.add_component("scorer", ArticleScorer, candidate_articles=e_candidates, interest_profile=e_user)
    n_ranker = builder.add_component("ranker", TopkRanker, {"num_slots": num_slots}, candidate_articles=n_scorer)

    # Fallback: sample from user topic interests
    n_topic_filter = builder.add_component(
        "topic-filter", TopicFilter, candidates=i_candidates, interest_profile=i_profile
    )
    n_sampler = builder.add_component("sampler", UniformSampler, candidates1=n_topic_filter, candidates2=i_candidates)

    # Combinei primary ranker and fallback
    builder.add_component("recommender", FillRecs, {"num_slots": num_slots}, recs1=n_ranker, recs2=n_sampler)
