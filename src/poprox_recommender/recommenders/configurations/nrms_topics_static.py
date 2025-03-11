# pyright: basic

from lenskit.pipeline import PipelineBuilder

from poprox_recommender.components.embedders import NRMSArticleEmbedder
from poprox_recommender.components.embedders.topic_wise_user import UserOnboardingEmbedder
from poprox_recommender.paths import model_file_path

from .. import common


def configure(builder: PipelineBuilder, num_slots: int, device: str):
    common.add_inputs(builder)
    common.add_article_embedder(
        builder, NRMSArticleEmbedder, model_path=model_file_path("nrms-mind/news_encoder.safetensors"), device=device
    )
    common.add_user_embedder(
        builder,
        UserOnboardingEmbedder,
        model_path=model_file_path("nrms-mind/user_encoder.safetensors"),
        device=device,
        embedding_source="static",
        topic_embedding="avg",
    )
    common.add_scorer(builder)
    common.add_rankers(builder, num_slots=num_slots)
    common.add_topic_fallback(builder, num_slots=num_slots)
