# pyright: basic
import logging
from os import PathLike
from typing import Protocol
from uuid import UUID

import torch as th
from safetensors.torch import load_file
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

from poprox_concepts import ArticleSet
from poprox_recommender.model import ModelConfig
from poprox_recommender.model.nrms.news_encoder import NewsEncoder
from poprox_recommender.paths import model_file_path
from poprox_recommender.pytorch.datachecks import assert_tensor_size
from poprox_recommender.pytorch.decorators import torch_inference

logger = logging.getLogger(__name__)
TITLE_LENGTH_LIMIT = 30


class ArticleEmbeddingModel(Protocol):
    """
    Interface exposed by article embedding models.
    """

    embedding_size: int

    def get_news_vector(self, news: th.Tensor) -> th.Tensor: ...


class NRMSArticleEmbedder:
    model: ArticleEmbeddingModel
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast
    device: str | None
    embedding_cache: dict[UUID, th.Tensor]

    def __init__(self, model_path: PathLike, device: str | None):
        checkpoint = load_file(model_path)

        config = ModelConfig()
        self.news_encoder = NewsEncoder(
            config.pretrained_model,
            config.num_attention_heads,
            config.additive_attn_hidden_dim,
        )
        self.news_encoder.load_state_dict(checkpoint)
        self.news_encoder.to(device)
        self.device = device

        plm_path = model_file_path(config.pretrained_model)
        logger.debug("loading tokenizer from %s", plm_path)

        self.tokenizer = AutoTokenizer.from_pretrained(plm_path, cache_dir="/tmp/")
        self.device = device
        self.embedding_cache = {}

    @torch_inference
    def __call__(self, article_set: ArticleSet) -> ArticleSet:
        if not article_set.articles:
            article_set.embeddings = th.zeros((0, self.news_encoder.embedding_size))  # type: ignore
            return article_set

        # Step 1: get the cached articles wherever possible.
        # Since Python dictionaries preserve order, this keeps the order aligned with the
        # input article set.
        cached = {article.article_id: self.embedding_cache.get(article.article_id) for article in article_set.articles}

        # Step 2: find the uncached articles.
        uncached = [article for article in article_set.articles if cached[article.article_id] is None]

        if uncached:
            logger.debug("need to embed %d of %d articles", len(uncached), len(cached))
            # Step 3: tokenize the uncached articles
            uc_title_tokens = th.stack(
                [
                    th.tensor(
                        self.tokenizer.encode(
                            article.title, padding="max_length", max_length=TITLE_LENGTH_LIMIT, truncation=True
                        ),
                        dtype=th.int32,
                    ).to(self.device)
                    for article in uncached
                ]
            )
            assert_tensor_size(uc_title_tokens, len(uncached), TITLE_LENGTH_LIMIT, label="uncached title tokens")

            # Step 4: embed the uncached articles
            uc_embeddings = self.news_encoder(uc_title_tokens)
            assert_tensor_size(
                uc_embeddings, len(uncached), self.news_encoder.plm_hidden_size, label="uncached article embeddings"
            )

            # Step 5: store embeddings to cache & result
            for i, uca in enumerate(uncached):
                # copy the tensor so it isn't attached to excess memory
                a_emb = uc_embeddings[i, :].clone()
                cached[uca.article_id] = a_emb
                self.embedding_cache[uca.article_id] = a_emb

        # Step 6: stack the embeddings into a single tensor
        # we do this with a list to properly deal with duplicate articles
        embed_single_tensors = [cached[article.article_id] for article in article_set.articles]  # type: ignore
        embed_tensor = th.stack(embed_single_tensors)  # type: ignore
        assert_tensor_size(
            embed_tensor, len(article_set.articles), self.news_encoder.plm_hidden_size, label="final article embeddings"
        )

        # Step 7: put the embedding tensor on the output
        article_set.embeddings = embed_tensor  # type: ignore

        return article_set
