import logging
from dataclasses import dataclass
from os import PathLike
from typing import List

import torch as th
from safetensors.torch import load_file
from transformers import AutoTokenizer

from poprox_recommender.model import ModelConfig
from poprox_recommender.model.nrms.news_encoder import NewsEncoder
from poprox_recommender.paths import model_file_path
from poprox_recommender.pytorch.decorators import torch_inference

logger = logging.getLogger(__name__)
CAPTION_LENGTH_LIMIT = 30


@dataclass
class CaptionEmbedderConfig:
    model_path: PathLike
    device: str | None


class CaptionEmbedder:
    config: CaptionEmbedderConfig
    model: NewsEncoder
    tokenizer: AutoTokenizer
    embedding_cache: dict[str, th.Tensor]

    def __init__(self, config: CaptionEmbedderConfig | None = None, **kwargs):
        if config is None:
            config = CaptionEmbedderConfig(
                model_path=model_file_path("nrms-mind/news_encoder.safetensors"), device="cpu"
            )

        self.config = config

        checkpoint = load_file(self.config.model_path)
        model_cfg = ModelConfig()
        self.model = NewsEncoder(
            model_file_path(model_cfg.pretrained_model),
            model_cfg.num_attention_heads,
            model_cfg.additive_attn_hidden_dim,
        )
        self.model.load_state_dict(checkpoint)
        self.model.to(self.config.device)

        plm_path = model_file_path(model_cfg.pretrained_model)
        logger.debug("loading tokenizer from %s", plm_path)
        self.tokenizer = AutoTokenizer.from_pretrained(plm_path, cache_dir="/tmp/", clean_up_tokenization_spaces=True)
        self.embedding_cache = {}

    @torch_inference
    def embed_captions(self, captions: List[str]) -> th.Tensor:
        if not captions:
            return th.zeros((0, self.model.embedding_size), device=self.config.device)

        cached = {caption: self.embedding_cache.get(caption) for caption in captions}
        uncached = [caption for caption in captions if cached[caption] is None]

        if uncached:
            logger.debug("need to embed %d of %d captions", len(uncached), len(captions))
            uc_caption_tokens = th.stack(
                [
                    th.tensor(
                        self.tokenizer.encode(
                            caption, padding="max_length", max_length=CAPTION_LENGTH_LIMIT, truncation=True
                        ),
                        dtype=th.int32,
                    ).to(self.config.device)
                    for caption in uncached
                ]
            )
            uc_embeddings = self.model(uc_caption_tokens)
            for i, caption in enumerate(uncached):
                a_emb = uc_embeddings[i, :].clone()
                cached[caption] = a_emb
                self.embedding_cache[caption] = a_emb

        embed_tensors = [cached[caption] for caption in captions]
        embed_tensor = th.stack(embed_tensors)
        return embed_tensor
