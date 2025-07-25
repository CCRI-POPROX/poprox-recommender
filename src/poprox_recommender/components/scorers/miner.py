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

@dataclass
class MinerArticleScorerConfig:
    model_path: PathLike
    device: str = "cpu"
    max_clicks_per_user: int = 50

class MinerArticleScorer(Component):
    config: MinerArticleScorerConfig

    def __init__(self, config: MinerArticleScorerConfig | None = None, **kwargs):
        super().__init__(config, **kwargs)
        checkpoint = load_file(self.config.model_path)

        roberta_config = RobertaConfig.from_pretrained("FacebookAI/roberta-base")

        newsEncoder = NewsEncoder(roberta_config,
    apply_reduce_dim: bool,
    use_sapo: bool,
    dropout: float,
    freeze_transformer: bool,)

        self.model = Miner(newsEncoder,
    use_category_bias: bool,
    num_context_codes: int,
    context_code_dim: int,
    score_type: str,
    dropout: float,
    num_category: int | None = None,
    category_embed_dim: int | None = None,
    category_pad_token_id: int | None = None,
    category_embed: Any | None = None)

        #model_cfg = ModelConfig()
        #self.user_encoder = UserEncoder(model_cfg.hidden_size, model_cfg.num_attention_heads)
        #self.user_encoder.load_state_dict(checkpoint)
        #self.user_encoder.to(self.config.device)

    @torch_inference
    def __call__(self, candidate_articles: CandidateSet, clicked_articles: CandidateSet, interest_profile: InterestProfile, ) -> CandidateSet:



        return candidate_articles
