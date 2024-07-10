# pyright: basic
import logging
from typing import Protocol
from uuid import UUID

import torch as th
from transformers import PreTrainedTokenizer

from poprox_concepts import ArticleSet
from poprox_recommender.torch.datachecks import assert_tensor_size
from poprox_recommender.torch.decorators import torch_inference

logger = logging.getLogger(__name__)
TITLE_LENGTH_LIMIT = 30


class ArticleEmbeddingModel(Protocol):
    """
    Interface exposed by article embedding models.
    """

    embedding_size: int

    def get_news_vector(self, news: th.Tensor) -> th.Tensor: ...


class ArticleEmbedder:
    model: ArticleEmbeddingModel
    tokenizer: PreTrainedTokenizer
    device: str | None
    embedding_cache: dict[UUID, th.Tensor]

    def __init__(self, model: ArticleEmbeddingModel, tokenizer: PreTrainedTokenizer, device: str | None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.embedding_cache = {}

    @torch_inference
    def __call__(self, article_set: ArticleSet) -> ArticleSet:
        if not article_set.articles:
            article_set.embeddings = th.zeros((0, self.model.embedding_size))  # type: ignore
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
            uc_embeddings = self.model.get_news_vector(uc_title_tokens)
            assert_tensor_size(
                uc_embeddings, len(uncached), self.model.embedding_size, label="uncached article embeddings"
            )

            # Step 5: store embeddings to cache & result
            for i, uca in enumerate(uncached):
                # copy the tensor so it isn't attached to excess memory
                a_emb = uc_embeddings[i, :].clone()
                cached[uca.article_id] = a_emb
                # use CPU for cached articles to reduce CUDA memory usage
                self.embedding_cache[uca.article_id] = a_emb.cpu()

        # Step 6: stack the embeddings into a single tensor
        # we do this with a list to properly deal with duplicate articles
        embed_single_tensors = [cached[article.article_id].to(self.device) for article in article_set.articles]  # type: ignore
        embed_tensor = th.stack(embed_single_tensors)  # type: ignore
        assert_tensor_size(
            embed_tensor, len(article_set.articles), self.model.embedding_size, label="final article embeddings"
        )

        # Step 7: put the embedding tensor on the output
        article_set.embeddings = embed_tensor  # type: ignore

        return article_set
