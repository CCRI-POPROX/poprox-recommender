import logging
from dataclasses import dataclass
from typing import Sequence

import torch as th
from transformers import AutoModel, AutoTokenizer

from poprox_recommender.components.embedders.article import (
    NRMSArticleEmbedder,
    NRMSArticleEmbedderConfig,  # Added import
)

logger = logging.getLogger(__name__)


@dataclass
class CaptionEmbedderConfig:
    model_path: str
    device: str


class CaptionEmbedder:
    def __init__(self, config: CaptionEmbedderConfig):
        self.config = config

        # Convert CaptionEmbedderConfig to NRMSArticleEmbedderConfig
        article_config = NRMSArticleEmbedderConfig(
            model_path=config.model_path,
            device=config.device,
        )
        self.article_embedder = NRMSArticleEmbedder(config=article_config)

        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.encoder = AutoModel.from_pretrained("distilbert-base-uncased").to(self.config.device)

    def __call__(self, captions: Sequence[str]) -> th.Tensor:
        tokens = self.tokenizer(
            captions,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.config.device)

        output = self.encoder(**tokens)
        caption_embeds = output.last_hidden_state[:, 0, :]

        return caption_embeds
