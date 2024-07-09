# pyright: basic
from typing import Protocol

import torch as th
from transformers import PreTrainedTokenizer

from poprox_concepts import ArticleSet

TITLE_LENGTH_LIMIT = 30


class ArticleEmbeddingModel(Protocol):
    """
    Interface exposed by article embedding models.
    """

    def get_news_vector(self, news: th.Tensor) -> th.Tensor: ...


class ArticleEmbedder:
    model: ArticleEmbeddingModel
    tokenizer: PreTrainedTokenizer
    device: str | None

    def __init__(self, model: ArticleEmbeddingModel, tokenizer: PreTrainedTokenizer, device: str | None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def __call__(self, article_set: ArticleSet) -> ArticleSet:
        tokenized_titles = [
            th.tensor(
                self.tokenizer.encode(
                    article.title, padding="max_length", max_length=TITLE_LENGTH_LIMIT, truncation=True
                ),
                dtype=th.int32,
            )
            for article in article_set.articles
        ]

        title_tensor = th.stack(tokenized_titles).to(self.device)
        assert title_tensor.shape == (len(article_set.articles), TITLE_LENGTH_LIMIT)

        article_set.embeddings = self.model.get_news_vector(title_tensor)  # type: ignore

        return article_set
