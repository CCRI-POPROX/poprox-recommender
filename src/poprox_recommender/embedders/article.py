# pyright: basic
from typing import Protocol

import torch as th
from transformers import PreTrainedTokenizer

from poprox_concepts import ArticleSet


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
        tokenized_titles = {}
        for article in article_set.articles:
            tokenized_titles[article.article_id] = self.tokenizer.encode(
                article.title, padding="max_length", max_length=30, truncation=True
            )

        title_tensor = th.tensor(list(tokenized_titles.values())).to(self.device)
        if len(title_tensor.shape) == 1:
            title_tensor = title_tensor.unsqueeze(dim=0)

        article_set.embeddings = self.model.get_news_vector(title_tensor)  # type: ignore

        return article_set
