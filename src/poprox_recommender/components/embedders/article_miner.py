# pyright: basic
import logging
from dataclasses import dataclass
from os import PathLike
from typing import Any

import torch as th
from lenskit.pipeline import Component

#from poprox_recommender.pytorch.datachecks import assert_tensor_size
from transformers import AutoConfig, AutoModel, AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

from poprox_concepts.domain import CandidateSet
from poprox_recommender.paths import model_file_path
from poprox_recommender.pytorch.decorators import torch_inference

logger = logging.getLogger(__name__)
TITLE_LENGTH_LIMIT = 30

#not efficient because you need to re-emebd the headline for every user

@dataclass
class MinerArticleEmbedderConfig:
    model_path: PathLike
    device: str | None

class MinerArticleEmbedder(Component):
    config: MinerArticleEmbedderConfig

    model: Any
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast

    def __init__(self, config: MinerArticleEmbedderConfig | None = None, **kwargs):
        super().__init__(config, **kwargs)

        plm_path = model_file_path(config.model_path)
        logger.debug("loading tokenizer from %s", plm_path)

        self.tokenizer = AutoTokenizer.from_pretrained(plm_path, cache_dir="/tmp/", clean_up_tokenization_spaces=True)

        self.plm_config = AutoConfig.from_pretrained(config.model_path, cache_dir="/tmp/")
        self.plm = AutoModel.from_config(self.plm_config)
        self.plm.requires_grad_(False)

    @torch_inference
    def __call__(self, article_set: CandidateSet) -> CandidateSet:
        embeddings = []

        for article in article_set.articles:
            inputs = self.tokenizer(article.headline, return_tensors="pt")
            outputs = self.plm(**inputs)
            embeddings.append(outputs)

        #Turns it from a list of vectors to a matrix
        article_set.embeddings = th.stack(embeddings)

        return article_set
