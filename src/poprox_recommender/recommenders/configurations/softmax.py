# pyright: basic

from lenskit.pipeline import PipelineBuilder

from poprox_recommender.components.embedders import NRMSArticleEmbedder, NRMSUserEmbedder
from poprox_recommender.components.samplers import SoftmaxSampler
from poprox_recommender.paths import model_file_path

from .. import common


def configure(builder: PipelineBuilder, num_slots: int, device: str):
    # standard practice is to put these calls in this order, to reuse logic
    common.add_inputs(builder)
    common.add_article_embedder(
        builder, NRMSArticleEmbedder, model_path=model_file_path("nrms-mind/news_encoder.safetensors"), device=device
    )
    common.add_user_embedder(
        builder, NRMSUserEmbedder, model_path=model_file_path("nrms-mind/user_encoder.safetensors"), device=device
    )
    common.add_scorer(builder)
    common.add_rankers(builder, SoftmaxSampler, num_slots=num_slots, temperature=30.0)
    common.add_topic_fallback(builder, num_slots=num_slots)
