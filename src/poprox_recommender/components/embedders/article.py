# pyright: basic
import logging
from dataclasses import dataclass
from os import PathLike
from typing import Protocol
from uuid import UUID

import torch as th
from lenskit.pipeline import Component
from safetensors.torch import load_file
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

from poprox_concepts.domain import CandidateSet
from poprox_recommender.model import ModelConfig
from poprox_recommender.model.nrms.news_encoder import NewsEncoder
from poprox_recommender.paths import model_file_path
from poprox_recommender.pytorch.datachecks import assert_tensor_size
from poprox_recommender.pytorch.decorators import torch_inference

logger = logging.getLogger(__name__)
TITLE_LENGTH_LIMIT = 30


@dataclass
class NRMSArticleEmbedderConfig:
    model_path: PathLike
    device: str | None


class ArticleEmbeddingModel(Protocol):
    """
    Interface exposed by article embedding models.
    """

    embedding_size: int

    def get_news_vector(self, news: th.Tensor) -> th.Tensor: ...


class NRMSArticleEmbedder(Component):
    config: NRMSArticleEmbedderConfig

    model: ArticleEmbeddingModel
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast
    embedding_cache: dict[UUID, th.Tensor]

    def __init__(self, config: NRMSArticleEmbedderConfig | None = None, **kwargs):
        super().__init__(config, **kwargs)

        checkpoint = load_file(self.config.model_path)
        model_cfg = ModelConfig()
        self.news_encoder = NewsEncoder(
            model_file_path(model_cfg.pretrained_model),
            model_cfg.num_attention_heads,
            model_cfg.additive_attn_hidden_dim,
        )
        self.news_encoder.load_state_dict(checkpoint)
        self.news_encoder.to(self.config.device)

        plm_path = model_file_path(model_cfg.pretrained_model)
        logger.debug("loading tokenizer from %s", plm_path)

        self.tokenizer = AutoTokenizer.from_pretrained(plm_path, cache_dir="/tmp/", clean_up_tokenization_spaces=True)
        self.embedding_cache = {}

    @torch_inference
    def __call__(self, article_set: CandidateSet) -> CandidateSet:
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
                            article.headline, padding="max_length", max_length=TITLE_LENGTH_LIMIT, truncation=True
                        ),
                        dtype=th.int32,
                    ).to(self.config.device)
                    for article in uncached
                ]
            )
            assert_tensor_size(uc_title_tokens, len(uncached), TITLE_LENGTH_LIMIT, label="uncached title tokens")

            # Step 4: embed the uncached articles
            uc_embeddings = self.news_encoder(uc_title_tokens)
            assert_tensor_size(
                uc_embeddings,
                len(uncached),
                self.news_encoder.plm_config.hidden_size,
                label="uncached article embeddings",
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
            embed_tensor,
            len(article_set.articles),
            self.news_encoder.plm_config.hidden_size,
            label="final article embeddings",
        )

        # Step 7: put the embedding tensor on the output
        article_set.embeddings = embed_tensor  # type: ignore

        return article_set


class EmbeddingCopier(Component):
    config: None

    @torch_inference
    def __call__(self, candidate_set: CandidateSet, selected_set: CandidateSet) -> CandidateSet:
        """
        Copies article embeddings from a candidate set to a set of selected/recommended articles

        Parameters
        ----------
        candidate_set : CandidateSet
            A set of candidate articles with the `.embeddings` property filled in
            (e.g. with ArticleEmbedder)
        selected_set : CandidateSet
            A set of selected or recommended articles chosen from `candidate_set`

        Returns
        -------
        CandidateSet
            selected_set with `.embeddings` set using the embeddings from `candidate_set`
        """
        candidate_article_ids = [article.article_id for article in candidate_set.articles]

        assert_tensor_size(
            candidate_set.embeddings,
            len(candidate_article_ids),
            candidate_set.embeddings.shape[1],
            label="candidate article embeddings",
        )

        indices = [candidate_article_ids.index(article.article_id) for article in selected_set.articles]
        selected_set.embeddings = candidate_set.embeddings[indices]

        assert_tensor_size(
            selected_set.embeddings,
            len(selected_set.articles),
            candidate_set.embeddings.shape[1],
            label="copied article embeddings",
        )

        return selected_set
