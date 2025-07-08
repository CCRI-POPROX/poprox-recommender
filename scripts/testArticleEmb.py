from os import PathLike

import torch as th
from safetensors.torch import load_file
from transformers import AutoTokenizer

from poprox_recommender.model import ModelConfig
from poprox_recommender.model.nrms.news_encoder import NewsEncoder
from poprox_recommender.paths import model_file_path
from poprox_recommender.pytorch.decorators import torch_inference

TITLE_LENGTH_LIMIT = 30


@torch_inference
def embed_article(
    headline, model_path: PathLike = model_file_path("nrms-mind/news_encoder.safetensors"), device: str = "cpu"
) -> th.Tensor:
    checkpoint = load_file(model_path)
    model_cfg = ModelConfig()
    news_encoder = NewsEncoder(
        model_file_path(model_cfg.pretrained_model),
        model_cfg.num_attention_heads,
        model_cfg.additive_attn_hidden_dim,
    )
    news_encoder.load_state_dict(checkpoint)
    news_encoder.to(device)
    news_encoder.eval()

    plm_path = model_file_path(model_cfg.pretrained_model)
    tokenizer = AutoTokenizer.from_pretrained(plm_path, cache_dir="/tmp/", clean_up_tokenization_spaces=True)

    # Tokenize and encode article
    article_tensor = th.tensor(
        tokenizer.encode(
            headline,
            padding="max_length",
            max_length=TITLE_LENGTH_LIMIT,
            truncation=True,  # type: ignore
        ),
        dtype=th.int32,
    ).to(device)

    # Get article embedding
    article_embedding = news_encoder(article_tensor)

    return article_embedding
