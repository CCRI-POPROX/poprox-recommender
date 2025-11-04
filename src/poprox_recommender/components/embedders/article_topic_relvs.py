# pyright: basic
from dataclasses import dataclass

import torch as th
import torch.nn.functional as F
from safetensors import safe_open

from poprox_concepts import CandidateSet
from poprox_recommender.components.embedders import NRMSArticleEmbedder, NRMSArticleEmbedderConfig
from poprox_recommender.paths import model_file_path
from poprox_recommender.pytorch.decorators import torch_inference

TITLE_LENGTH_LIMIT = 30
MAX_ATTRIBUTES = 15


def load_topic_tensor_map(path: str, device: str = "cpu"):
    topic_map: dict[str, th.Tensor] = {}

    with safe_open(model_file_path(path), framework="pt", device="cpu") as f:
        for key in f.keys():
            topic_map[key] = F.normalize(f.get_tensor(key).to(device), dim=0)
    return topic_map


def extract_topical_embeddings(article, topic_map_1):
    mention_names = {m.entity.name for m in article.mentions if m.entity is not None}
    return [topic_map_1[m] for m in mention_names if m in topic_map_1]


def extract_topical_embeddings_hybrid(article, topic_map_1, topic_map_2):
    mention_names = {m.entity.name for m in article.mentions if m.entity is not None}
    hybrid_tensors = []
    for m in mention_names:
        if m in topic_map_1 and m in topic_map_2:
            hybrid = 0.5 * topic_map_1[m] + 0.5 * topic_map_2[m]
            hybrid_tensors.append(hybrid)

    return hybrid_tensors


@dataclass
class NRMSArticleTopicEmbedder(NRMSArticleEmbedder):
    config: NRMSArticleEmbedderConfig

    def __init__(self, config: NRMSArticleEmbedderConfig | None = None, **kwargs):
        super().__init__(config, **kwargs)

        # Please make sure the the user topic embedder have the same prelearned embedder file as input

        # self.topic_map_1 = load_topic_tensor_map(path="topic_embeddings_cand_11_months.safetensors")
        self.topic_map_2 = None  # if not hybrid please set self.topic_map_2 = none  # noqa: E501
        self.topic_map_1 = load_topic_tensor_map(path="topic_embeddings_def_llm.safetensors")

    @torch_inference
    def __call__(self, article_set: CandidateSet) -> CandidateSet:
        if not article_set.articles:
            article_set.embeddings = th.zeros((0, self.news_encoder.embedding_size))  # type: ignore
            return article_set

        all_article_embeddings: list[th.Tensor] = []

        for article in article_set.articles:
            headline_tensors = th.tensor(
                self.tokenizer.encode(
                    article.headline, padding="max_length", max_length=TITLE_LENGTH_LIMIT, truncation=True
                ),
                dtype=th.int32,
                device="cpu",
            ).unsqueeze(0)

            headline_embedding = self.news_encoder(headline_tensors).squeeze(0)
            headline_embedding = F.normalize(headline_embedding, dim=0)

            if self.topic_map_2 is not None:
                topic_tensors = extract_topical_embeddings_hybrid(article, self.topic_map_1, self.topic_map_2)
            else:
                topic_tensors = extract_topical_embeddings(article, self.topic_map_1)

            num_topic = len(topic_tensors)

            if num_topic > 0:
                topic_embeddings = th.stack(topic_tensors, dim=0)
                topic_embedding_mean = F.normalize(topic_embeddings.mean(dim=0), dim=0)
                final_embedding = 0.5 * headline_embedding + 0.5 * topic_embedding_mean
            else:
                final_embedding = headline_embedding

            all_article_embeddings.append(final_embedding)

        embed_tensor = th.stack(all_article_embeddings, dim=0)
        article_set.embeddings = embed_tensor  # type: ignore

        return article_set
