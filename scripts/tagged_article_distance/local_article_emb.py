from os import PathLike
from typing import Tuple

import torch as th
from safetensors.torch import load_file
from transformers import AutoTokenizer, PreTrainedTokenizer

from poprox_recommender.model import ModelConfig
from poprox_recommender.model.nrms.news_encoder import NewsEncoder
from poprox_recommender.paths import model_file_path
from poprox_recommender.pytorch.decorators import torch_inference

TITLE_LENGTH_LIMIT = 30


def load_news_encoder(
    model_path: PathLike = model_file_path("nrms-mind/news_encoder.safetensors"),
    device: str = "cpu",
) -> Tuple[NewsEncoder, PreTrainedTokenizer]:
    # Load model checkpoint
    checkpoint = load_file(model_path)

    # Create model config
    model_cfg = ModelConfig()

    # Initialize encoder
    news_encoder = NewsEncoder(
        model_file_path(model_cfg.pretrained_model),
        model_cfg.num_attention_heads,
        model_cfg.additive_attn_hidden_dim,
    )
    news_encoder.load_state_dict(checkpoint)
    news_encoder.to(device)
    news_encoder.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_file_path(model_cfg.pretrained_model),
        cache_dir="/tmp/",
        clean_up_tokenization_spaces=True,
    )

    return news_encoder, tokenizer


@torch_inference
def embed_article(
    headline: str,
    model: NewsEncoder,
    tokenizer: PreTrainedTokenizer,
    device: str = "cpu",
) -> th.Tensor:
    article_encode = tokenizer.encode(
        headline,
        padding="max_length",
        max_length=TITLE_LENGTH_LIMIT,
        truncation=True,
    )

    article_tensor = th.tensor([article_encode], dtype=th.int32).to(device)
    embedding = model(article_tensor).squeeze(0)
    return embedding
