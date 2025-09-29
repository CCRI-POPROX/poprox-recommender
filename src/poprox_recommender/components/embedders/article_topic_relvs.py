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
    mention_relevance_pairs = {(m.entity.name, m.relevance) for m in article.mentions if m.entity is not None}

    topical_texts = []
    for mention, score in mention_relevance_pairs:
        # if mention in topic_descriptions and score >= 76:
        if mention in topic_descriptions:
            topical_texts.append(topic_descriptions[mention])

    return topical_texts


@dataclass
class NRMSArticleTopicEmbedder(NRMSArticleEmbedder):
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
            topical_text = find_topical_text(TOPIC_DESCRIPTIONS, article)
            num_topic = len(topical_text)
            attribute_text.extend(topical_text)

            while len(attribute_text) < MAX_ATTRIBUTES:
                attribute_text.append("")
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

            headline_embedding = attribute_embeddings[0]

            if num_topic > 0:
                topic_embeddings = attribute_embeddings[1 : num_topic + 1]
                topic_embedding = topic_embeddings.mean(dim=0)
                final_embedding = 0.5 * headline_embedding + 0.5 * topic_embedding
            else:
                final_embedding = headline_embedding

            all_article_embeddings.append(final_embedding)

        embed_tensor = th.stack(all_article_embeddings)

        article_set.embeddings = embed_tensor  # type: ignore

        # breakpoint()
        return article_set
