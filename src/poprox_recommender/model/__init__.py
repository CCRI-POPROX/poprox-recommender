from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from safetensors.torch import load_file
from transformers import AutoTokenizer

from poprox_recommender.config import default_device
from poprox_recommender.model.nrms import NRMS
from poprox_recommender.paths import model_file_path

LANG_MODEL_NAME = "distilbert-base-uncased"
_cached_model: Optional[RecommenderComponents] = None
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    num_epochs: float = 10

    num_clicked_news_a_user: float = 50
    word_freq_threshold: float = 1
    dropout_probability: float = 0.2
    word_embedding_dim: float = 300
    category_embedding_dim: float = 100
    query_vector_dim: float = 200
    additive_attn_hidden_dim: float = 200
    num_attention_heads: float = 16
    hidden_size: int = 768

    pretrained_model = LANG_MODEL_NAME


@dataclass
class RecommenderComponents:
    tokenizer: AutoTokenizer
    model: NRMS
    device: str | None


def load_checkpoint(device=None):
    checkpoint = None
    if device is None:
        device = default_device()

    load_path = model_file_path("model.safetensors")
    logger.debug("loading model checkpoint from %s", load_path)

    checkpoint = load_file(load_path)
    return checkpoint, device


def load_model(checkpoint, device):
    logger.debug("instantiating NRMS model")
    model = NRMS(ModelConfig()).to(device)
    model.load_state_dict(checkpoint)
    model.eval()

    return model
