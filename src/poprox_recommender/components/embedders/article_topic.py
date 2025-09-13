# pyright: basic
import logging

import torch as th

from poprox_recommender.components.embedders.article import TITLE_LENGTH_LIMIT, NRMSArticleEmbedder

logger = logging.getLogger(__name__)


class NRMSArticleTopicEmbedder(NRMSArticleEmbedder):
    def _tokenize_articles(self, articles):
        tokenized = [
            th.tensor(
                self.tokenizer.encode(
                    f"{', '.join([mention.entity.name for mention in article.mentions])}: {article.headline}",
                    padding="max_length",
                    max_length=TITLE_LENGTH_LIMIT,
                    truncation=True,
                ),
                dtype=th.int32,
            ).to(self.config.device)
            for article in articles
        ]

        return th.stack(tokenized)
