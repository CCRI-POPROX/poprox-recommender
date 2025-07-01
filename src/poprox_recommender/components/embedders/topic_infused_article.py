# pyright: basic
from dataclasses import dataclass

import torch as th
import torch.nn.functional as F

from poprox_concepts import CandidateSet
from poprox_recommender.components.embedders import NRMSArticleEmbedder, NRMSArticleEmbedderConfig
from poprox_recommender.components.topical_description import TOPIC_DESCRIPTIONS
from poprox_recommender.pytorch.decorators import torch_inference

TITLE_LENGTH_LIMIT = 30
MAX_ATTRIBUTES = 15


def find_topical_text(topic_descriptions, article):
    unique_mention = {mention.entity.name for mention in article.mentions}
    topical_texts = []
    for mention in unique_mention:
        if mention in topic_descriptions:
            topical_texts.append(topic_descriptions[mention])
    return topical_texts


@dataclass
class FMStyleArticleEmbedder(NRMSArticleEmbedder):
    config: NRMSArticleEmbedderConfig

    def __init__(self, config: NRMSArticleEmbedderConfig | None = None, **kwargs):
        super().__init__(config, **kwargs)

    @torch_inference
    def __call__(self, article_set: CandidateSet) -> CandidateSet:
        if not article_set.articles:
            article_set.embeddings = th.zeros((0, self.news_encoder.embedding_size))  # type: ignore
            return article_set

        all_article_embeddings = []

        for article in article_set.articles:
            attribute_text = [article.headline]

            # this part is for topical embedddings, remove this for topicless embeddings
            # topical_text = find_topical_text(TOPIC_DESCRIPTIONS, article)
            # num_topic = len(topical_text)
            # attribute_text.extend(topical_text)

            # while len(attribute_text) < MAX_ATTRIBUTES:
            #     attribute_text.append("")
            # this part is for topical embedddings, remove this for topicless embeddings

            attribute_tensors = th.stack(
                [
                    th.tensor(
                        self.tokenizer.encode(
                            text, padding="max_length", max_length=TITLE_LENGTH_LIMIT, truncation=True
                        ),
                        dtype=th.int32,
                    ).to(self.config.device)
                    for text in attribute_text
                ]
            )

            attribute_embeddings = self.news_encoder(attribute_tensors)

            attribute_embeddings = F.normalize(attribute_embeddings, dim=1)

            # attribute_embeddings[1 : num_topic + 1] = attribute_embeddings[1 : num_topic + 1] / num_topic #for topicless embedding, remove this line

            all_article_embeddings.append(attribute_embeddings)

        embed_tensor = th.stack(all_article_embeddings)

        article_set.embeddings = embed_tensor  # type: ignore

        return article_set
