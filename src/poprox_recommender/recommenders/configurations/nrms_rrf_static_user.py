# pyright: basic

from lenskit.pipeline import PipelineBuilder

from poprox_recommender.components.embedders import NRMSArticleEmbedder
from poprox_recommender.components.embedders.topic_wise_user import UserOnboardingEmbedder
from poprox_recommender.components.embedders.user import NRMSUserEmbedder
from poprox_recommender.components.joiners import ReciprocalRankFusion
from poprox_recommender.components.rankers.topk import TopkRanker
from poprox_recommender.components.scorers import ArticleScorer
from poprox_recommender.paths import model_file_path

from .. import common


def configure(builder: PipelineBuilder, num_slots: int, device: str):
    common.add_inputs(builder)
    common.add_article_embedder(
        builder, NRMSArticleEmbedder, model_path=model_file_path("nrms-mind/news_encoder.safetensors"), device=device
    )
    common.add_user_embedder(
        builder,
        NRMSUserEmbedder,
        model_path=model_file_path("nrms-mind/user_encoder.safetensors"),
        device=device,
    )
    common.add_scorer(builder)
    o_rank_1 = common.add_rankers(builder, num_slots=num_slots, recommender=False)

    # Second user embedding strategy
    e_user_2 = builder.add_component(
        "user-embedder2",
        UserOnboardingEmbedder,
        {
            "model_path": model_file_path("nrms-mind/user_encoder.safetensors"),
            "device": device,
            "embedding_source": "static",
            "topic_embedding": "avg",
        },
        candidate_articles=builder.node("candidate"),
        clicked_articles=builder.node("history-embedder"),
        interest_profile=builder.node("profile"),
    )
    o_scored_2 = builder.add_component(
        "scorer2", ArticleScorer, candidate_articles=builder.node("candidate-embedder"), interest_profile=e_user_2
    )
    o_rank_2 = builder.add_component("ranker2", TopkRanker, {"num_slots": num_slots}, candidate_articles=o_scored_2)

    builder.add_component("recommender", ReciprocalRankFusion, {"num_slots": num_slots}, recs1=o_rank_1, recs2=o_rank_2)
