from dataclasses import dataclass
from os import PathLike

import torch
from lenskit.pipeline import Component
from safetensors.torch import load_file

from poprox_concepts import CandidateSet, InterestProfile
from poprox_recommender.pytorch.decorators import torch_inference
from poprox_recommender.model.miner.model import Miner
from poprox_recommender.model.miner.news_encoder import NewsEncoder
from transformers import RobertaConfig

from safetensors.torch import load_model
from transformers import RobertaConfig


@dataclass
class MinerArticleScorerConfig:
    model_path: PathLike
    device: str = "cpu"
    max_clicks_per_user: int = 50

class MinerArticleScorer(Component):
    config: MinerArticleScorerConfig

    def __init__(self, config: MinerArticleScorerConfig | None = None, **kwargs):
    #     super().__init__(config, **kwargs)
    #     checkpoint = load_file(self.config.model_path)

        config = RobertaConfig.from_pretrained("FacebookAI/roberta-base")

        news_encoder = NewsEncoder.from_pretrained("FacebookAI/roberta-base", config=config,
            apply_reduce_dim=True, use_sapo=True,
            dropout=0.2, freeze_transformer=True,
            word_embed_dim=256, combine_type="linear")

        model = Miner(news_encoder=news_encoder, use_category_bias=False, num_context_codes=32, context_code_dim=200, score_type="weighted", dropout=0.2)

        print("Loading model weights...")
        loaded = load_model(model, "miner_2025-07-18.safetensors")
        print("Done")

    @torch_inference
    def __call__(self, candidate_articles: CandidateSet, clicked_articles: CandidateSet, interest_profile: InterestProfile, ) -> CandidateSet:



        return candidate_articles
